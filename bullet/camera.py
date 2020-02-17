import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation as R
from typing import *

import bullet.util


class BulletCamera:
    def __init__(self, offset=[0.0, 0.0, 0.0]):
        """
        Args:
            p: The pybullet client.
            offset: The position offset to apply.

        Attributes:
            p: The pybullet client.
            H: The height of the generated images.
            W: The width of the generated images.
            position: The position of the camera, in world coordinate frame.
            rotation: The rotation of the camera (in degrees), in world 
                coordinate frame.
            offset: The position offset to apply.
            cam_target_pos: The target position of the camera.
            up_vector: The up vector of the camera.
            view_mat: The view matrix of the camera.
            proj_mat: The projection matrix of the camera.
        """
        self.p = bullet.util.create_bullet_client(mode="direct")

        # Camera configurations.
        self.H = 480  # image dim
        self.W = 320

        self.position = None
        self.rotation = None
        self.offset = offset

        self.cam_target_pos = [0.0, 0.0, 0.0]
        self.up_vector = [0.0, 0.0, 1.0]

        self.view_mat = None
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

    def set_pose(self, position: List[float], rotation: List[float]):
        """Sets the camera pose, including all the camera attributes related to
        pose.

        Args:
            position: The xyz position of the camera.
            rotation: The xyz rotation of the camera.
        """
        # Apply to offset to the camera position.
        position = self.offset_in_direction(
            source=position, offset=self.offset, rotation=rotation
        )

        self.position = position
        self.rotation = rotation
        self.target_position = self.compute_target_position(
            position=position, rotation=rotation
        )
        self.up_vector = self.compute_up_vector(rotation=rotation)
        self.view_mat = self.compute_view_matrix()

    def compute_view_matrix(self) -> List[float]:
        """Computes the view matrix based on the camera attributes."""
        view_mat = self.p.computeViewMatrix(
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

    def get_rgb_and_mask(self, p: bc.BulletClient):
        view_mat = np.array(self.view_mat).reshape((4, 4))
        img = p.getCameraImage(
            self.H,
            self.W,
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.reshape(img[2], (self.W, self.H, 4))[:, :, :3]
        mask = np.reshape(img[4], (self.W, self.H))
        return rgb, mask

    def to_json(self) -> Dict:
        json_dict = {
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
        json_dict: The JSON dictionary.
    
    Returns:
        camera: BulletCamera.
    """
    camera = BulletCamera()
    camera.set_pose(
        position=json_dict["position"], rotation=json_dict["rotation"]
    )
    return camera
