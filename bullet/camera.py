import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc


class BulletCamera:
    def __init__(self, p=None):
        self.p = (
            bc.BulletClient(connection_mode=pybullet.DIRECT)
            if p is None
            else p
        )

        # Camera configurations.
        self.H = 480  # image dim
        self.W = 320
        self.cam_target_pos = [0.25, 0.2, 0]  # [0, 0, 0]
        self.cam_distance = 0.81
        self.pitch = -30.0
        self.roll = 0
        self.yaw = 270
        self.up_axis_index = 2
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

        self.view_mat = self.p.computeViewMatrixFromYawPitchRoll(
            self.cam_target_pos,
            self.cam_distance,
            self.yaw,
            self.pitch,
            self.roll,
            self.up_axis_index,
        )

    def set_cam_transform(self, position):
        transform = np.eye(4)
        transform[0:3, -1] = position
        self.view_mat = transform.T
        # print(transform)

    def set_cam_position(
        self, eye_position, target_position, up_vec=[0.0, 0.0, 1.0]
    ):
        self.view_mat = self.p.computeViewMatrix(
            cameraEyePosition=eye_position,
            cameraTargetPosition=target_position,
            cameraUpVector=up_vec,
        )

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
        # assert np.min(mask) == -1
        return rgb, mask
