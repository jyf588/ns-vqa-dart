import math
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Optional, Tuple


class DashObject:
    def __init__(
        self,
        shape: str,
        size: str,
        color: str,
        world_position: List[int],
        world_orientation: List[int],
    ):
        self.shape = shape
        self.size = size
        self.color = color
        self.world_position = world_position
        self.world_orientation = world_orientation


class DashTable(DashObject):
    def __init__(self):
        self.shape = "tabletop"
        self.size = None
        self.color = "grey"
        self.world_position = [0.25, 0.0, 0.0]  # [0.25, 0.2, 0.0]
        self.world_orientation = [0.0, 0.0, 0.0, 1.0]


class DashRobot:
    def __init__(self, p):
        self.p = p
        self.robot_id = self.render_robot()

        self.axis2joint_name = {
            "pan": "head_pan_joint",
            "tilt": "head_tilt_joint",
            "roll": "head_roll_joint",
        }
        self.cam_position_joint_name = "eyes_pan_joint"
        self.cam_offset = [0.02, 0.0, 0.0]
        # self.cam_offset = [0.0, 0.0, 0.0]

        self.joint_name2id = self.get_joint_name2id()

        # Initialize the head and camera to zero rotation.
        self.set_head_and_camera(pan=0, tilt=50, roll=0, degrees=True)

    def render_robot(self) -> int:
        """Renders the robot in bullet.

        Returns:
            robot_id: The unique bullet ID of the robot.
        """
        robot_id = self.p.loadURDF(
            "/Users/michelleguo/workspace/third_party/jyf588/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_revolute_head.urdf",
            basePosition=[-0.3, 0.0, -1.25],
        )
        return robot_id

    def get_joint_name2id(self) -> Dict[str, int]:
        joint_name2id = {}
        for joint_idx in range(self.p.getNumJoints(self.robot_id)):
            joint_info = self.p.getJointInfo(self.robot_id, joint_idx)
            joint_id = joint_info[0]
            joint_name = joint_info[1].decode("utf-8")
            joint_name2id[joint_name] = joint_id
        return joint_name2id

    def set_head_and_camera(
        self, pan: int, tilt: int, roll: int, degrees=True
    ) -> Tuple[List]:
        self.set_head_rotation(pan=pan, tilt=tilt, roll=roll, degrees=degrees)
        self.cam_position, self.cam_target_position, self.cam_up_vector = self.get_camera_pose(
            pan=pan, tilt=tilt, roll=roll, degrees=degrees
        )

    def set_head_rotation(
        self, pan: int, tilt: int, roll: int, degrees: Optional[bool] = True
    ):
        """Sets the rotation of the head.
        
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

    def get_camera_pose(
        self, pan: int, tilt: int, roll: int, degrees: Optional[bool] = True
    ):
        rotation_xyz = [roll, tilt, pan]

        # Offset the camera position.
        camera_position = self.get_camera_position()
        camera_position = self.offset_in_direction(
            source_xyz=camera_position,
            offset=self.cam_offset,
            rotation_xyz=rotation_xyz,
            degrees=degrees,
        )

        camera_target_position = self.offset_in_direction(
            source_xyz=camera_position,
            offset=[1.0, 0.0, 0.0],
            rotation_xyz=rotation_xyz,
            degrees=degrees,
        )

        camera_up_vector = self.rotate_vector(
            vector=[0.0, 0.0, 1.0], rotation_xyz=rotation_xyz, degrees=degrees
        )
        camera_up_vector[1] *= -1
        return camera_position, camera_target_position, camera_up_vector

    def get_camera_position(self) -> np.ndarray:
        joint_id = self.joint_name2id[self.cam_position_joint_name]
        position = self.p.getLinkState(self.robot_id, joint_id)[0]
        return position

    def offset_in_direction(
        self,
        source_xyz: List[float],
        offset: List[float],
        rotation_xyz: List[float],
        degrees: Optional[bool] = True,
    ) -> List[float]:
        vector = self.rotate_vector(
            vector=offset, rotation_xyz=rotation_xyz, degrees=degrees
        )
        new_xyz = source_xyz + vector
        return list(new_xyz)

    def rotate_vector(
        self,
        vector: List[float],
        rotation_xyz: List[float],
        degrees: Optional[bool] = True,
    ):
        rot = R.from_euler("xyz", rotation_xyz, degrees=degrees)
        vector = rot.apply(vector)
        return vector
