import cv2
import math
import numpy as np
import os
import pybullet_utils.bullet_client as bc
import pycocotools.mask as mask_util
from typing import *
import sys

# sys.path.append("./")
# from .camera import BulletCamera
# from . import util

import bullet2unity.states
from ns_vqa_dart.bullet.camera import BulletCamera
from ns_vqa_dart.bullet import util

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
            y += self.construct_attr_vec(shape=self.shape, color=self.color)

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
        return to_caption(json_dict=self.to_json())


def to_caption(json_dict: Dict):
    str_list = []
    # Add up vector if it doesn't already exist.
    if "up_vector" not in json_dict:
        json_dict["up_vector"] = util.orientation_to_up(
            orientation=json_dict["orientation"]
        )

    for k, v in json_dict.items():
        # Convert to cm and set precision to 1 decimal point.
        if k in ["radius", "height", "position"]:
            if type(v) == tuple:
                v = list(v)
            if type(v) == list:
                v = [float(f"{v_i * 100:.1f}") for v_i in v]
            elif type(v) == float:
                v = f"{v * 100:.1f}"
            k = f"{k} (cm)"
        # Set precision to 1 decimal point.
        elif k in ["orientation", "up_vector"]:
            v = [float(f"{v_i:.1f}") for v_i in v]
        # elif k in ["img_id", "orientation", "oid"]:
        #     continue
        str_list.append(f"{k}: {v}")
    return str_list


def compute_X(
    oid: int,
    img: np.ndarray,
    seg: np.ndarray,
    keep_occluded: bool,
    data_height: Optional[int] = 480,
    data_width: Optional[int] = 480,
) -> Optional[np.ndarray]:
    """Constructs the data tensor for an object.

    Args:
        oid: The ID of the object that we are generating data for.
        img: An image of a scene.
        seg: A 2D segmentation map where each pixel stores the object 
            ID it belongs to.
        keep_occluded: Whether to keep objects that are completely occluded.
        data_height: The desired height of the generated data tensor.
        data_width: The desired width of the generated data tensor.
    
    Returns:
        data (np.ndarray, type=np.uint8): The final data, which contains two 
            concatenated images:
            (1) An image of the object only, cropped using its segmentation
                mask, and maximally enlarged such that the aspect ratio of the
                object is maintained. Shape (`data_height`, `data_width`, 3).
            (2) The original image of the scene, with the segmentation area of
                the object cropped out. Shape (`data_height`, `data_width`, 3).

            Note: If the object bbox area is zero, then we return one of two
            things, depending on the value of `keep_occluded`:
                (1) True: input RGB is simply the original RGB image, and the 
                    object segmentation image is all zeros.
                (2) False: We simply return None as an error message.
    """
    # Verify that the RGB image and the segmentation have proper dimensions.
    img_H, img_W, C = img.shape
    seg_H, seg_W = seg.shape
    assert img_H == seg_H
    assert img_W == seg_W

    # Initialize the generated data tensor, which we will fill with the
    # appropriate data.
    data = np.zeros((data_height, data_width, 6)).astype(np.uint8)

    # Compute the object's bbox.
    bbox = compute_bbox(oid=oid, mask=seg)

    if bbox is None and not keep_occluded:
        return None

    # Generate the object's input image.
    input_object_img = compute_input_object_img(
        oid=oid,
        img=img,
        seg=seg,
        bbox=bbox,
        data_height=data_height,
        data_width=data_width,
    )

    # Generate the modified scene image, where object pixels are zeroed out.
    input_scene_img = compute_input_scene_img(
        oid=oid, img=img, seg=seg, H=data_height, W=data_width
    )

    # Assign the images to the final data tensor.
    data[:, :, :3] = input_object_img
    data[:, :, 3:6] = input_scene_img
    return data


def compute_input_scene_img(
    oid: int, img: np.ndarray, seg: np.ndarray, H: int, W: int
) -> np.ndarray:
    """Generate the modified input scene, where the object's pixels are zeroed
    out. Note that if the object has zero pixels in the segmentation, the input
    image is simply the full, original scene.

    Args:
        oid: The ID of the object that we are generating data for.
        img: An image of the scene.
        segmentation: A 2D segmentation map where each pixel stores the object 
            ID it belongs to.
        H: The desired height of the generated data tensor.
        W: The desired width of the generated data tensor.
    
    Returns:
        input_img: The generated input image for the scene.
    """
    # Initialize a tensor of specified size for the image we will generate.
    input_img = np.zeros((H, W, 3), dtype=np.uint8)

    # Zero out the object's pixels from the scene.
    img[seg == oid] = 0.0

    # Compute the offset of the original image's dimensions compared to the
    # desired output size. If original image is smaller, boundaries are padded
    # equally on all edges.
    img_H, img_W, _ = img.shape
    y_offset = (H - img_H) // 2
    x_offset = (W - img_W) // 2
    assert y_offset >= 0
    assert x_offset >= 0
    input_img[y_offset : img_H + y_offset, x_offset : img_W + x_offset] = img
    return input_img


def compute_input_object_img(
    oid: int,
    img: np.ndarray,
    seg: np.ndarray,
    bbox: List[float],
    data_height: int,
    data_width: int,
):
    """Generates the input object image.

    Args:
        oid: The ID of the object that we are generating data for.
        img: An image of the scene.
        segmentation: A 2D segmentation map where each pixel stores the object 
            ID it belongs to.
        bbox: The 2D bounding box of the object: (x, y, w, h)
        data_height: The desired height of the generated data tensor.
        data_width: The desired width of the generated data tensor.
    
    Returns:
        input_object_img: The generated input image for the object. If the 
            object does not have any pixels in the segmentation, the returned
            image is all zeros.
    """
    # Initialize the tensor that we will modify and return to the user.
    input_object_img = np.zeros((data_height, data_width, 3), dtype=np.uint8)

    # First, zero out all pixels outside of the object's segmentation area.
    img_without_background = img.copy()
    img_without_background[seg != oid] = 0.0

    # If bbox is None, this means that no pixels contain the object, so we are
    # done. We return an all-zero image.
    if bbox is None:
        print(f"Bbox is None. Object ID: {oid}")
        return input_object_img

    x, y, w, h = bbox
    seg = img_without_background[y : y + h, x : x + w, :]

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
    top_pad = (data_height - H_) // 2
    left_pad = (data_width - W_) // 2

    input_object_img[top_pad : top_pad + H_, left_pad : left_pad + W_] = seg
    return input_object_img


def compute_y(
    odict: Dict,
    coordinate_frame: str,
    camera: Optional[BulletCamera] = None,
    cam_position: List[float] = None,
    cam_orientation: List[float] = None,
) -> np.ndarray:
    """Constructs the label vector for an object.

    Args:
        odict: A dictionary of object labels with the format: 
            {
                "shape": <shape>,
                "color": <color>,
                "radius": <radius>,
                "height": <height>,
                "position": <position>,
                "orientation": <orientation
            }
        coordinate_frame: The coordinate frame to use, either "world" or
            "camera" coordinate frame.
        camera: The BulletCamera for computing values in camera coordinate 
            frame.
    
    Returns:
        y: Labels for the example, a 1-D array with the following structure:
            np.ndarray([
                <one_hot_shape>,
                <one_hot_color>,
                <radius>,
                <height>,
                <x_pos>,
                <y_pos>,
                <z_pos>,
                <up_vector[0]>,
                <up_vector[1]>,
                <up_vector[2]>,
            ])
    """
    y = []
    y += construct_attr_vec(shape=odict["shape"], color=odict["color"])
    y += [odict["radius"], odict["height"]]

    position = odict["position"]
    orientation = odict["orientation"]
    up_vector = util.orientation_to_up(orientation)

    if coordinate_frame == "camera":
        position = util.world_to_cam(xyz=position, camera=camera)
        up_vector = util.world_to_cam(xyz=up_vector, camera=camera)
    elif coordinate_frame == "unity_camera":
        position, up_vector = bullet2unity.states.bworld2ucam(
            p_bw=position,
            up_bw=up_vector,
            uworld_cam_position=cam_position,
            uworld_cam_orientation=cam_orientation,
        )
    elif coordinate_frame == "world":
        pass
    else:
        raise ValueError(f"Invalid coordinate frame: {coordinate_frame}.")

    y += position
    y += up_vector

    return np.array(y, dtype=np.float32)


def construct_attr_vec(shape: str, color: str) -> List[int]:
    """Constructs the attributes vector.
    
    Args:
        shape: The shape of the object.
        color: The color of the object.

    Returns:
        attr_vec: The attributes vector, which is a binary vector that 
            assigns 1 to owned attributes and 0 everywhere else.
    """
    attr_vec = np.zeros((len(ATTR2IDX),))
    for attr in [shape, color]:
        shape_idx = ATTR2IDX[attr]
        attr_vec[shape_idx] = 1
    return list(attr_vec)


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
    cam_position: Optional[List[float]] = None,
    cam_orientation: Optional[List[float]] = None,
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
    if coordinate_frame == "camera":
        if camera is None:
            euler_angles = util.orientation_to_euler(
                orientation=cam_orientation
            )
            camera = BulletCamera(position=cam_position, rotation=euler_angles)

        # raise ValueError(
        #     f"Coordinate frame is camera but no camera was provided."
        # )

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
    elif coordinate_frame == "unity_camera":
        (
            y_dict["position"],
            y_dict["up_vector"],
        ) = bullet2unity.states.ucam2bworld(
            p_uc=y_dict["position"],
            up_uc=y_dict["up_vector"],
            uworld_cam_position=cam_position,
            uworld_cam_orientation=cam_orientation,
        )
    else:
        raise ValueError(f"Invalid coordinate frame: {coordinate_frame}.")
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
    orientation = util.up_to_orientation(
        up=y_dict["up_vector"], gt_orientation=gt_orientation
    )

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


def assign_ids_to_odicts(odicts: List[Dict], start_id: int):
    """Assigns object IDs to object dictionaries.

    Args:
        odicts: A list of object dictionaries, with the format:
            [
                {
                    <attr>: <value>
                }
            ]
        start_id: The starting ID to assign object IDs.
    
    Returns:
        oid2odict: A mapping from assigned object ID to dictionary, with the 
            format: {
                <oid>: {
                    <attr>: <value>
                }
            }
    """
    next_id_to_assn = start_id
    oid2odict = {}
    for odict in odicts:
        oid = next_id_to_assn
        oid2odict[oid] = odict
        next_id_to_assn += 1
    return oid2odict


class DashTable(DashObject):
    def __init__(
        self,
        position: Optional[List[float]] = [0.25, 0.0, 0.0],
        offset: Optional[List[float]] = None,
    ):
        self.shape = "tabletop"
        self.size = None
        self.color = "grey"
        self.position = position  # [0.25, 0.2, 0.0]
        self.orientation = [0.0, 0.0, 0.0, 1.0]

        if offset is not None:
            self.position = list(np.array(self.position) + np.array(offset))


class DashRobot:
    def __init__(
        self,
        p: bc.BulletClient,
        urdf_path: str = "bullet/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_revolute_head.urdf",
        position: List[float] = [-0.3, 0.0, -1.25],
        include_head: bool = True,
        head_roll_joint_name: str = "head_roll_joint",
        head_tilt_joint_name: str = "head_tilt_joint",
        head_pan_joint_name: str = "head_pan_joint",
        cam_position_joint_name: str = "eyes_pan_joint",
        cam_directed_offset: List[float] = [0.02, 0.0, 0.0],
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
            cam_directed_offset: The offset to add to the camera joint as the 
                final camera position (e.g., if the joint is within the eye, we
                want to offset it such that it's at the surface of the eye).
        """
        self.p = p
        self.urdf_path = urdf_path
        self.position = position
        self.robot_id = self.load_robot()

        self.joint_name2id = self.get_joint_name2id()

        # The robot's head camera.
        if include_head:
            self.camera = BulletCamera(
                p=p, directed_offset=cam_directed_offset
            )

            self.axis2joint_name = {
                "roll": head_roll_joint_name,
                "tilt": head_tilt_joint_name,
                "pan": head_pan_joint_name,
            }
            self.cam_position_joint_name = cam_position_joint_name

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

    def set_state(self, state: Dict[str, float]):
        """Sets the robot state.

        Args:
            state: The robot state.
        """
        for joint_name, joint_angle in state.items():
            joint_id = self.joint_name2id[joint_name]
            self.p.resetJointState(self.robot_id, joint_id, joint_angle)
