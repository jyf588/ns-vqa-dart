import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation as R
from typing import *


class BulletCamera:
    def __init__(self, p=None, offset=[0.0, 0.0, 0.0]):
        self.p = (
            bc.BulletClient(connection_mode=pybullet.DIRECT)
            if p is None
            else p
        )

        # Camera configurations.
        self.H = 480  # image dim
        self.W = 320

        self.position = None
        self.rotation = None
        self.offset = offset

        self.cam_target_pos = [0.0, 0.0, 0.0]
        self.up_vector = [0.0, 0.0, 1.0]

        # Rotation
        # self.cam_distance = 0.81
        # self.pitch = -30.0
        # self.roll = 0
        # self.yaw = 270
        # self.up_axis_index = 2
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

        # self.view_mat = self.p.computeViewMatrixFromYawPitchRoll(
        #     self.cam_target_pos,
        #     self.cam_distance,
        #     self.yaw,
        #     self.pitch,
        #     self.roll,
        #     self.up_axis_index,
        # )

    def set_pose(
        self, position: List[float], rotation: List[float], degrees: bool
    ):
        """Sets the camera pose, including all the camera attributes related to
        pose.

        Args:
            position: The xyz position of the camera.
            rotation: The xyz rotation of the camera.
        """
        # Apply to offset to the camera position.
        position = self.offset_in_direction(
            source=position,
            offset=self.offset,
            rotation=rotation,
            degrees=degrees,
        )

        self.position = position
        self.rotation = rotation
        self.target_position = self.compute_target_position(
            position=position, rotation=rotation, degrees=degrees
        )
        self.up_vector = self.compute_up_vector(
            rotation=rotation, degrees=degrees
        )
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
        self, position: List[float], rotation: List[float], degrees: bool
    ) -> List[float]:
        """Computes the target position based on the camera position and the
        camera rotation.
        """
        target_position = self.offset_in_direction(
            source=position,
            offset=[1.0, 0.0, 0.0],
            rotation=rotation,
            degrees=degrees,
        )
        return target_position

    def compute_up_vector(
        self, rotation: List[float], degrees: bool
    ) -> List[float]:
        up_vector = self.rotate_vector(
            vector=[0.0, 0.0, 1.0], rotation=rotation, degrees=degrees
        )
        up_vector[1] *= -1
        return up_vector

    def offset_in_direction(
        self,
        source: List[float],
        offset: List[float],
        rotation: List[float],
        degrees: bool,
    ) -> List[float]:
        source = np.array(source)
        vector = np.array(
            self.rotate_vector(
                vector=offset, rotation=rotation, degrees=degrees
            )
        )
        new_xyz = source + vector
        return list(new_xyz)

    def rotate_vector(
        self, vector: List[float], rotation: List[float], degrees: bool
    ) -> List[float]:
        rot = R.from_euler("xyz", rotation, degrees=degrees)
        vector = rot.apply(vector)
        return list(vector)

    def get_rgb_and_mask(self):
        view_mat = np.array(self.view_mat).reshape((4, 4))
        img = self.p.getCameraImage(
            self.H,
            self.W,
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat,
            renderer=self.p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.reshape(img[2], (self.W, self.H, 4))[:, :, :3]
        mask = np.reshape(img[4], (self.W, self.H))
        return rgb, mask

    def to_json_dict(self) -> Dict:
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
