"""A class that tracks objects and saves states."""
import os
import pybullet as p
from typing import *

from . import util


class StateSaver:
    def __init__(self, dataset_dir: str):
        """
        Args:
            dataset_dir: The directory to save the states in.
        """
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)

        self.oid2attr = {}
        self.robot_id = None

        # A counter for example IDs.
        self.i = 0

    def reset(self):
        """Clears the list of tracked objects."""
        self.oid2attr = {}

    def set_robot_id(self, robot_id: int):
        self.robot_id = robot_id

    def track_object(self, oid: int, shape: str, radius: float, height: float):
        """Stores an object's metadata for tracking.
        
        Args:
            oid: The object ID.
            shape: The shape of the object.
            radius: The radius of the object.
            height: The height of the object.
        """
        self.oid2attr[oid] = {
            "shape": shape,
            "radius": radius,
            "height": height,
        }

    def save(self):
        """Saves the current state of the bullet scene for tracked objects."""
        state = self.get_current_state()
        path = os.path.join(self.dataset_dir, f"{self.i:06}.p")
        util.save_pickle(path=path, data=state)
        self.i += 1

    def get_current_state(self):
        """Update the state (position and orientation) of objects we are
        tracking.

        Note that position is based on the COM of the object.
        """
        state = {"objects": [], "robot": {}}

        # Get object states.
        for oid, odict in self.oid2attr.items():
            position, orientation = p.getBasePositionAndOrientation(oid)
            odict["position"] = list(position)
            odict["orientation"] = list(orientation)
            state["objects"].append(odict)

        # Get robot state.
        for joint_idx in range(p.getNumJoints(self.robot_id)):
            joint_name = p.getJointInfo(self.robot_id, joint_idx)[1].decode(
                "utf-8"
            )
            joint_angle = p.getJointState(
                bodyUniqueId=self.robot_id, jointIndex=joint_idx
            )[0]
            state["robot"][joint_name] = joint_angle
        return state
