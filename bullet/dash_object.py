import pybullet_utils.bullet_client as bc
import math
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple

from bullet.camera import BulletCamera


class DashObject:
    def __init__(
        self,
        shape: str,
        size: str,
        color: str,
        position: List[float],
        orientation: List[float],
    ):
        """
        Attributes:
            shape: The shape of the object.
            size: The size of the object.
            color: The color of the object.
            position: The xyz position of the object, in world coordinate 
                frame.
            orientation: The orientation of the object, expressed as a 
                [x, y, z, w] quaternion, in world coordinate frame.            
        """
        self.id = None
        self.shape = shape
        self.size = size
        self.color = color
        self.position = position
        self.orientation = orientation

    def set_id(self, oid: int):
        self.id = oid

    def to_json(self) -> Dict[str, Any]:
        json_dict = {
            "id": self.id,
            "shape": self.shape,
            "size": self.size,
            "color": self.color,
            "position": self.position,
            "orientation": self.orientation,
        }
        return json_dict


def from_json(json_dict: Dict) -> DashObject:
    o = DashObject(
        shape=json_dict["shape"],
        size=json_dict["size"],
        color=json_dict["color"],
        position=json_dict["position"],
        orientation=json_dict["orientation"],
    )
    return o


class DashTable(DashObject):
    def __init__(self):
        self.shape = "tabletop"
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
        self.robot_id = self.render_robot()

        # The robot's head camera.
        self.camera = BulletCamera(p=p, offset=cam_offset)

        self.axis2joint_name = {
            "roll": head_roll_joint_name,
            "tilt": head_tilt_joint_name,
            "pan": head_pan_joint_name,
        }
        self.cam_position_joint_name = cam_position_joint_name
        self.joint_name2id = self.get_joint_name2id()

        # Initialize the head and camera to zero rotation.
        self.set_head_and_camera(roll=0, tilt=0, pan=0, degrees=True)

    def render_robot(self) -> int:
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
            position=self.get_camera_position(),
            rotation=[roll, tilt, pan],
            degrees=degrees,
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
