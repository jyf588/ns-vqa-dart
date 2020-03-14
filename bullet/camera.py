import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation as R
from typing import *
import sys

# sys.path.append("./")
# from . import util
from ns_vqa_dart.bullet import util


class BulletCamera:
    def __init__(
        self,
        p: Optional[bc.BulletClient] = None,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        offset: Optional[List[float]] = None,
        directed_offset: Optional[List[float]] = None,
        init_type: Optional[str] = None,
    ):
        """
        Args:
            p: The pybullet client containing the scenes that we want to take
                snapshots of.
            position: The position of the camera, in world coordinate frame.
            rotation: The rotation of the camera (in degrees), in world 
                coordinate frame.
            offset: The amount of offset to apply to the camera position.
            directed_offset: The position offset to apply in the direction of 
                the camera forward vector.
            init_type: The type of camera to initialize with.

        Attributes:
            p: The pybullet client containing the scenes that we want to take
                snapshots of.
            H: The height of the generated images.
            W: The width of the generated images.
            position: The position of the camera, in world coordinate frame.
            rotation: The rotation of the camera (in degrees), in world 
                coordinate frame.
            offset: The amount of offset to apply to the camera position.
            directed_offset: The position offset to apply in the direction of 
                the camera up vector.
            cam_target_pos: The target position of the camera.
            up_vector: The up vector of the camera.
            view_mat: The view matrix of the camera.
            proj_mat: The projection matrix of the camera.
        """
        self.p = p
        # self.p_util = bullet.util.create_bullet_client(mode="direct")

        # Camera configurations.
        self.H = 480  # image dim
        self.W = 320

        self.position = position
        self.rotation = rotation
        self.offset = offset
        self.directed_offset = directed_offset
        if position is not None:
            self.set_pose(position=position, rotation=rotation)

        # self.target_position = [0.0, 0.0, 0.0]
        # self.up_vector = [0.0, 0.0, 1.0]
        # self.view_mat = None

        self.proj_mat = [
            1.0825318098068237,
            0.0,
            0.0,
            0.0,
            0.0,
            1.732050895690918,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0002000331878662,
            -1.0,
            0.0,
            0.0,
            -0.020002000033855438,
            0.0,
        ]

        self.init_type = init_type
        if init_type == "default":
            self.setup_default_camera()
        elif init_type == "z_axis":
            self.setup_axis_camera(axis="z")

    def set_bullet_client(self, p: bc.BulletClient):
        """Sets the bullet client of the camera.

        Args:
            p: The bullet client.
        """
        self.p = p

    def setup_default_camera(self):
        """Sets the camera to the default pose, which is behind the robot arm
        and centered.
        """
        self.target_position = [0.25, 0.0, 0.0]
        self.view_mat = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target_position,
            distance=0.81,
            yaw=270.0,
            pitch=-30.0,
            roll=0.0,
            upAxisIndex=2,
        )

    def setup_axis_camera(self, axis: str):
        """Sets the camera to be axis-aligned with a user-specified axis.

        Args:
            axis: The axis to align with.
        """
        if axis != "z":
            raise NotImplementedError
        self.target_position = [0.25, 0.0, 0.0]
        self.view_mat = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target_position,
            distance=0.81,
            yaw=270.0,
            pitch=0.0,
            roll=0.0,
            upAxisIndex=2,
        )

    def set_pose(self, position: List[float], rotation: List[float]):
        """Sets the camera pose, including all the camera attributes related to
        pose.

        Args:
            position: The xyz position of the camera.
            rotation: The xyz rotation of the camera.
        """
        # print(f"set_pose / before offset / position: {position}")
        # print(f"set_pose / before offset / offset: {self.offset}")

        if self.offset is not None:
            position = list(np.array(position) + np.array(self.offset))

        # print(f"set_pose / before directed_offset / position: {position}")
        # print(
        #     f"set_pose / before directed_offset / directed_offset: {self.directed_offset}"
        # )

        # Apply to offset to the camera position.
        if self.directed_offset is not None:
            position = self.offset_in_direction(
                source=position, offset=self.directed_offset, rotation=rotation
            )

        self.position = position
        self.rotation = rotation

        # print(f"set_pose / position: {self.position}")
        # print(f"set_pose / rotation: {self.rotation}")

        self.target_position = self.compute_target_position(
            position=position, rotation=rotation
        )
        self.up_vector = self.compute_up_vector(rotation=rotation)
        self.view_mat = self.compute_view_matrix()

    def compute_view_matrix(self) -> List[float]:
        """Computes the view matrix based on the camera attributes."""
        # print(f"compute_view_matrix / position: {self.position}")
        # print(f"compute_view_matrix / target_position: {self.target_position}")
        # print(f"compute_view_matrix / up_vector: {self.up_vector}")
        view_mat = pybullet.computeViewMatrix(
            cameraEyePosition=self.position,
            cameraTargetPosition=self.target_position,
            cameraUpVector=self.up_vector,
        )
        return view_mat

    def compute_target_position(
        self, position: List[float], rotation: List[float]
    ) -> List[float]:
        """Computes the target position based on the camera position and the
        camera rotation.
        """
        target_position = self.offset_in_direction(
            source=position, offset=[1.0, 0.0, 0.0], rotation=rotation
        )
        return target_position

    def compute_up_vector(self, rotation: List[float]) -> List[float]:
        up_vector = self.rotate_vector(
            vector=[0.0, 0.0, 1.0], rotation=rotation
        )
        up_vector[1] *= -1
        return up_vector

    def offset_in_direction(
        self, source: List[float], offset: List[float], rotation: List[float]
    ) -> List[float]:
        source = np.array(source)
        vector = np.array(self.rotate_vector(vector=offset, rotation=rotation))
        new_xyz = source + vector
        return list(new_xyz)

    def rotate_vector(
        self, vector: List[float], rotation: List[float]
    ) -> List[float]:
        rot = R.from_euler("xyz", rotation, degrees=True)
        vector = rot.apply(vector)
        return list(vector)

    def get_rgb_and_mask(self):
        img = self.p.getCameraImage(
            self.H,
            self.W,
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        # print(f"get_rgb_and_mask / self.view_mat: {self.view_mat}")
        rgb = np.reshape(img[2], (self.W, self.H, 4))[:, :, :3]
        mask = np.reshape(img[4], (self.W, self.H))
        return rgb, mask

    def to_json(self) -> Dict:
        json_dict = {
            "init_type": self.init_type,
            "H": self.H,
            "W": self.W,
            "position": self.position,
            "rotation": self.rotation,
            "offset": self.offset,
            "target_position": self.target_position,
            "up_vector": self.up_vector,
            "view_mat": self.view_mat,
            "proj_mat": self.proj_mat,
        }
        return json_dict


def from_json(json_dict: Dict) -> BulletCamera:
    """Create a BulletCamera from a JSON dictionary.

    Note that the offset from the JSON dictionary is not used in the 
    BulletCamera constructed here, because the position in the JSON dictionary
    already includes the offset.

    Args:
        p: The BulletClient to use.
        json_dict: The JSON dictionary.
    
    Returns:
        camera: BulletCamera.
    """
    if "init_type" in json_dict:
        init_type = json_dict["init_type"]
    else:
        init_type = None
    camera = BulletCamera(init_type=init_type)
    if init_type is None:
        camera.set_pose(
            position=json_dict["position"], rotation=json_dict["rotation"]
        )
    return camera
