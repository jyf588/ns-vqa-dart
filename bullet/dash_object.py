import cv2
import math
import numpy as np
import os
import pybullet_utils.bullet_client as bc
import pycocotools.mask as mask_util
from typing import Any, Dict, List, Optional, Tuple

from .camera import BulletCamera
from . import util


# Note: order matters!
ATTR_NAME2LIST = {
    "shape": ["box", "cylinder", "sphere"],
    "color": ["red", "yellow", "green", "blue"],
}
ATTR2IDX = {}
idx = 0
for name, attr_list in ATTR_NAME2LIST.items():
    for label in attr_list:
        ATTR2IDX[label] = idx
        idx += 1


class DashObject:
    def __init__(
        self,
        shape: str,
        color: str,
        radius: float,
        height: float,
        position: List[float],
        orientation: Optional[List[float]] = [0.0, 0.0, 0.0, 1.0],
        oid: Optional[int] = None,
        img_id: Optional[int] = None,
    ):
        """
        Args:
            shape: The shape of the object.
            color: The color of the object.
            radius: The radius of the object.
            height: The height of the object.
            position: The xyz position of the center of the object's base, in 
                world coordinate frame.
            orientation: The orientation of the object, expressed as a 
                [x, y, z, w] quaternion, in world coordinate frame.     
            oid: PyBullet object ID.
            img_id: The image ID associated with the object.      

        Attributes:
            shape: The shape of the object.
            color: The color of the object.
            radius: The radius of the object.
            height: The height of the object.
            position: The xyz position of the center of the object's base, in 
                world coordinate frame.
            orientation: The orientation of the object, expressed as a 
                [x, y, z, w] quaternion, in world coordinate frame.     
            oid: PyBullet object ID that comes from p.loadURDF.
            img_id: The image ID associated with the object.      
        """
        assert len(position) == 3

        self.shape = shape
        self.color = color
        self.radius = radius
        self.height = height
        self.position = position
        self.orientation = orientation
        self.oid = oid
        self.img_id = img_id

    def mask_area(self, mask: np.ndarray) -> int:
        """Computes the object mask area.

        Args:
            mask: The mask for all objects in a scene.
        
        Returns:
            area: The number of mask pixels that the object has.
        """
        rle = compute_object_mask(oid=self.oid, mask=mask)
        area = mask_util.area(rle)
        return area

    def to_y_vec(
        self,
        use_attr: bool,
        use_size: bool,
        use_position: bool,
        use_up_vector: bool,
        coordinate_frame: str,
        camera: BulletCamera,
    ) -> np.ndarray:
        """Constructs the label vector for an object.

        Args:
            use_attr: Whether to include attributes in the label.
            use_size: Whether to include the size (radius, height) in the 
                label.
            use_position: Whether to include position in the label.
            use_up_vector: Whether to include the up vector in the label.
            use_height: Whether to include height in the label.
            coordinate_frame: The coordinate frame to use, either "world" or
                "camera" coordinate frame.
            camera: The BulletCamera for computing values in camera coordinate 
                frame.
        
        Returns:
            y: Labels for the example.
        """
        y = []
        if use_attr:
            y += list(self.construct_attr_vec())

        if use_size:
            y += list([self.radius, self.height])

        if coordinate_frame == "world":
            if use_position:
                y += self.position
            if use_up_vector:
                y += self.compute_up_vector()
        elif coordinate_frame == "camera":
            if use_position:
                y += util.world_to_cam(xyz=self.position, camera=camera)
            if use_up_vector:
                y += util.world_to_cam(
                    xyz=self.compute_up_vector(), camera=camera
                )
        else:
            raise ValueError(f"Invalid coordinate frame: {coordinate_frame}.")
        return np.array(y)

    def construct_attr_vec(self) -> np.ndarray:
        """Constructs the attributes vector.
        
        Returns:
            attr_vec: The attributes vector, which is a binary vector that 
                assigns 1 to owned attributes and 0 everywhere else.
        """
        attr_vec = np.zeros((len(ATTR2IDX),))
        for attr in [self.shape, self.color]:
            shape_idx = ATTR2IDX[attr]
            attr_vec[shape_idx] = 1
        return attr_vec

    def compute_up_vector(self) -> List[float]:
        """Computes the up vector for the current object.

        Returns:
            up_vector: The up vector for the object.
        """
        up_vector = util.orientation_to_up(self.orientation)
        return up_vector

    def to_json(self) -> Dict[str, Any]:
        """Converts the current DashObject into a JSON dictionary.

        Returns:
            json_dict: The JSON dictionary representing the DashObject.
        """
        json_dict = {
            "oid": self.oid,
            "img_id": self.img_id,
            "shape": self.shape,
            "color": self.color,
            "radius": self.radius,
            "height": self.height,
            "position": self.position,
            "orientation": self.orientation,
            "up_vector": self.compute_up_vector(),
        }
        return json_dict

    def to_caption(self) -> List[str]:
        """Converts the DashObject into a list of strings. Useful for OpenCV
        captioning.

        Returns:
            str_list: A list of strings.
        """
        str_list = []
        json_dict = self.to_json()
        for k, v in json_dict.items():
            if k in ["radius", "height", "position"]:
                if type(v) == list:
                    v = [float(f"{v_i * 100:.1f}") for v_i in v]
                elif type(v) == float:
                    v = f"{v * 100:.1f}"
                k = f"{k} (cm)"
            if k in ["img_id", "orientation", "up_vector", "oid"]:
                continue
            str_list.append(f"{k}: {v}")
        return str_list


def compute_data_from_rgb_and_mask(
    oid: int,
    rgb: np.ndarray,
    mask: np.ndarray,
    data_height: Optional[int] = 480,
    data_width: Optional[int] = 480,
) -> np.ndarray:
    """Constructs the data tensor for an object.

    Args:
        oid: The object ID.
        rgb: The RGB image of the entire scene.
        mask: A 2D mask where each pixel holds the object ID it belongs to.
        data_height: The height of the data tensor.
        data_width: The width of the data tensor.
    
    Returns:
        data: The final data, which contains a cropped image of the object
            concatenated with the original image of the scene, with the
            object cropped out. (RGB, HWC)

            Note: If the object bbox area is zero, the input RGB is simply the
            original RGB image, and the object segmentation image is all zeros.
    """
    rgb = rgb.copy()
    bbox = compute_bbox(oid=oid, mask=mask)
    data = np.zeros((data_height, data_width, 6)).astype(np.uint8)
    input_rgb = rgb.copy()
    if bbox is None:
        print(f"Bbox is None. Object ID: {oid}")
    else:
        x, y, w, h = bbox

        # Set the object seg to zeros in the original RGB image.
        input_rgb[mask == oid] = 0.0

        # Create the segmentation image (maintain aspect ratio, use
        # replicate padding).
        # First, zero out everything in the RGB image except for the object
        # segmentation.
        rgb_with_only_seg = rgb.copy()
        rgb_with_only_seg[mask != oid] = 0.0

        # Crop the segmentation out using its bbox.
        seg = rgb_with_only_seg[y : y + h, x : x + w, :]

        # Compute the new dimensions to resize the segmentation to.
        if h > w:
            aspect_ratio = h / w
            resize_dims = (data_height, int(data_width / aspect_ratio))
        else:
            aspect_ratio = w / h
            resize_dims = (int(data_height / aspect_ratio), data_width)
        H_, W_ = resize_dims

        # Resize the segmentation while maintaining aspect ratio.
        seg = cv2.resize(seg, (W_, H_))  # OpenCV expects WH.
        seg_padded = np.zeros((data_height, data_width, 3), dtype=np.uint8)
        top_pad = (data_height - H_) // 2
        left_pad = (data_width - W_) // 2
        seg_padded[top_pad : top_pad + H_, left_pad : left_pad + W_] = seg
        data[:, :, :3] = seg_padded
    data[80:400, :, 3:6] = input_rgb
    return data


def compute_bbox(oid: int, mask: np.ndarray) -> Tuple[int]:
    """Compute the object bounding box.

    Args:
        oid: The object ID.
        mask: A 2D mask where values represent the object ID of each pixel.
    
    Returns:
        (x, y, w, h): The bounding box of the object.
    """
    rle = compute_object_mask(oid=oid, mask=mask)

    if mask_util.area(rle) > 0:
        bbox = mask_util.toBbox(rle)  # xywh

        # Convert to integers.
        x, y, w, h = [int(bbox[i]) for i in (0, 1, 2, 3)]
        return x, y, w, h
    else:
        return None


def compute_object_mask(oid: int, mask: np.ndarray) -> List[int]:
    """Computes the object mask, in RLE format.

    Args:
        oid: The object ID.
        mask: The mask of all objects in a scene.
    
    Returns:
        rle: The object mask, in RLE format.
    """
    # Binary mask of the object.
    obj_mask = mask == oid

    # Mask to bbox.
    mask = np.asfortranarray(obj_mask, dtype="uint8")
    rle = mask_util.encode(mask)
    rle["counts"] = rle["counts"].decode("ASCII")
    return rle


def y_vec_to_dict(
    y: List[float],
    coordinate_frame: str,
    camera: Optional[BulletCamera] = None,
) -> Dict:
    """Converts a y vector containing object labels into dictionary. Values are
    converted back into global coordinate frame if `coordinate_frame` is 
    `camera`.

    Args:
        y: A vector of labels.
        coordinate_frame: The coordinate frame to use, either "world" or
                "camera" coordinate frame.
        camera: The BulletCamera for computing values in camera coordinate 
            frame. Required if the coordinate frame is "camera".

    Returns:
        y_dict: A dictionary representation of the y vector. All xyz values are
            in world coordinate frame. Contains the following keys: {
                "shape": str,
                "color": str,
                "size": str,
                "radius": float,
                "height": float,
                "position": [float, float, float],
                "up_vector": [float, float, float]
            }
    """
    assert type(y) == list
    if coordinate_frame == "camera" and camera is None:
        raise ValueError(
            f"Coordinate frame is camera but no camera was provided."
        )

    y_dict = {}
    start = 0
    for name, attr_list in ATTR_NAME2LIST.items():
        end = start + len(attr_list)
        attr_idx = np.argmax(y[start:end])
        start = end
        label = attr_list[attr_idx]
        y_dict[name] = label

    y_dict["radius"] = y[start]
    start += 1
    y_dict["height"] = y[start]
    start += 1
    end = start + 3
    y_dict["position"] = y[start:end]
    start = end
    end = start + 3
    y_dict["up_vector"] = y[start:end]

    # Converts from camera to world coordinate frame if necessary.
    if coordinate_frame == "world":
        pass
    elif coordinate_frame == "camera":
        for k in ["position", "up_vector"]:
            y_dict[k] = util.cam_to_world(xyz=y_dict[k], camera=camera)
    else:
        raise ValueError(f"Invalid coordinate frame: {coordinate_frame}.")

    # Assigns a categorical size label for the given radius and height.
    if y_dict["radius"] > 0.04 and y_dict["height"] > 0.18:
        size = "large"
    else:
        size = "small"
    y_dict["size"] = size
    return y_dict


def y_dict_to_object(
    y_dict: Dict,
    img_id: Optional[int] = None,
    oid: Optional[int] = None,
    gt_orientation: Optional[List[float]] = None,
) -> DashObject:
    """Converts a y dictionary to a DashObject.

    Args:
        y_dict: The y dictionary.
        gt_orientation: The GT orientation. If supplied, the GT z rotation 
            (i.e., the first two columns of the rotation matrix), is included
            in the orientation attribute of the DashObject.
    
    Returns:
        o: The corresponding DashObject.
    """
    if gt_orientation is None:
        print(f"Warning: GT orientation is not supplied.")
        rotation = util.orientation_to_rotation(orientation=gt_orientation)
        rotation = np.array(rotation).reshape((3, 3))
    else:
        rotation = np.zeros((3, 3))

    # Set the up vector.
    rotation[:, -1] = y_dict["up_vector"]

    # Convert to orientation.
    orientation = util.rotation_to_quaternion(rotation=rotation)

    o = DashObject(
        img_id=img_id,
        oid=oid,
        shape=y_dict["shape"],
        color=y_dict["color"],
        radius=y_dict["radius"],
        height=y_dict["height"],
        position=y_dict["position"],
        orientation=orientation,
    )
    return o


def from_json(json_dict: Dict) -> DashObject:
    o = DashObject(
        oid=json_dict["oid"],
        img_id=json_dict["img_id"],
        shape=json_dict["shape"],
        color=json_dict["color"],
        radius=json_dict["radius"],
        height=json_dict["height"],
        position=json_dict["position"],
        orientation=json_dict["orientation"],
    )
    return o


class DashTable(DashObject):
    def __init__(self):
        self.shape = "table"
        self.size = None
        self.color = "grey"
        self.position = [0.25, 0.0, 0.0]  # [0.25, 0.2, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]


class DashRobot:
    def __init__(
        self,
        p: bc.BulletClient,
        urdf_path: str = "bullet/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_revolute_head.urdf",
        position: List[float] = [-0.3, 0.0, -1.25],
        head_roll_joint_name: str = "head_roll_joint",
        head_tilt_joint_name: str = "head_tilt_joint",
        head_pan_joint_name: str = "head_pan_joint",
        cam_position_joint_name: str = "eyes_pan_joint",
        cam_offset: List[float] = [0.02, 0.0, 0.0],
    ):
        """
        Args:
            p: The PyBullet client.
            urdf_path: The path to the urdf file of the robot.
            position: The base position of the robot.
            head_roll_joint_name: The joint representing head roll.
            head_tilt_joint_name: The joint representing head tilt.
            head_pan_joint_name: The joint representing head pan.
            cam_position_joint_name: The joint representing the camera position.
            cam_offset: The offset to add to the camera joint as the final 
                camera position (e.g., if the joint is within the eye, we want
                to offset it such that it's at the surface of the eye).
        """
        self.p = p
        self.urdf_path = urdf_path
        self.position = position
        self.robot_id = self.load_robot()

        # The robot's head camera.
        self.camera = BulletCamera(offset=cam_offset)

        self.axis2joint_name = {
            "roll": head_roll_joint_name,
            "tilt": head_tilt_joint_name,
            "pan": head_pan_joint_name,
        }
        self.cam_position_joint_name = cam_position_joint_name
        self.joint_name2id = self.get_joint_name2id()

        # Initialize the head and camera to zero rotation.
        self.set_head_and_camera(roll=0, tilt=0, pan=0, degrees=True)

    def load_robot(self) -> int:
        """Renders the robot in bullet.

        Returns:
            robot_id: The unique bullet ID of the robot.
        """
        robot_id = self.p.loadURDF(self.urdf_path, basePosition=self.position)
        return robot_id

    def get_joint_name2id(self) -> Dict[str, int]:
        """Constructs a mapping from joint names to joint IDs.

        Returns:
            joint_name2id: The dictionary mapping joint name to IDs.
        """
        joint_name2id = {}
        for joint_idx in range(self.p.getNumJoints(self.robot_id)):
            joint_info = self.p.getJointInfo(self.robot_id, joint_idx)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("utf-8")
            joint_name2id[joint_name] = joint_id
        return joint_name2id

    def set_head_and_camera(
        self, roll: int, tilt: int, pan: int, degrees=True
    ) -> Tuple[List]:
        """Sets the head and camera pose.

        Args:
            roll: The roll angle.
            tilt: The tilt angle.
            pan: The pan angle.
            degrees: If true, angles are degrees, Otherwise, angles are radians.
        """
        self.set_head_rotation(roll=roll, tilt=tilt, pan=pan, degrees=degrees)
        self.camera.set_pose(
            position=self.get_camera_position(), rotation=[roll, tilt, pan]
        )

    def set_head_rotation(
        self, roll: int, tilt: int, pan: int, degrees: Optional[bool] = True
    ):
        """Sets the rotation of the robot head.
        
        Args:
            pan: The pan rotation of the head (i.e., look left/right).
            tilt: The tilt rotation of the head (i.e., look up/down).
            roll: The roll rotation of the head (i.e., ear-to-shoulders).
            degrees: If true, assumes the input rotation is in degrees. 
                Otherwise, assumes the input rotation is in radians.
        """
        axis2rot = {"pan": pan, "tilt": tilt, "roll": roll}

        for axis, joint_name in self.axis2joint_name.items():
            joint_id = self.joint_name2id[joint_name]
            rot = axis2rot[axis]

            # Convert to radians because bullet expects radians.
            if degrees:
                rot = math.radians(rot)

            # Set the rotation for the head joint.
            self.p.resetJointState(self.robot_id, joint_id, rot)

    def get_camera_position(self) -> np.ndarray:
        """Get the position of the camera based the joint associated with the
        camera.
        """
        joint_id = self.joint_name2id[self.cam_position_joint_name]
        position = self.p.getLinkState(self.robot_id, joint_id)[0]
        return position
