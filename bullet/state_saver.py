"""A class that tracks objects and saves states."""
import copy
import os
from typing import *

from . import util
from . import random_objects


class StateSaver:
    def __init__(self, p, dataset_dir: Optional[str] = None):
        """
        Args:
            dataset_dir: The directory to save the states in.
        """
        self.p = p
        self.dataset_dir = dataset_dir
        if self.dataset_dir is not None:
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

    def track_object(
        self,
        oid: int,
        shape: str,
        radius: float,
        height: float,
        color: Optional[str] = None,
    ):
        """Stores an object's metadata for tracking.
        
        Args:
            oid: The object ID.
            shape: The shape of the object.
            radius: The radius of the object.
            height: The height of the object.
            color: The color of the object. If not set, a color is randomly
                chosen.
        """
        if color is None:
            color = random_objects.generate_random_color()
        self.oid2attr[oid] = {
            "shape": shape,
            "color": color,
            "radius": radius,
            "height": height,
        }

    def save(self):
        """Saves the current state of the bullet scene for tracked objects."""
        if self.dataset_dir is None:
            raise ValueError(f"self.dataset_dir is None.")
        state = self.get_current_state()
        path = os.path.join(self.dataset_dir, f"{self.i:06}.p")
        util.save_pickle(path=path, data=state)
        self.i += 1

    def get_current_state(self):
        """Update the state (position and orientation) of objects we are
        tracking.

        Note that position is based on the COM of the object.

        Returns:
            state: A dictionary of the scene state. Expected format:
                {
                    "objects": [{<attr>: <value>}]
                    "robot": {<joint_name>: <joint_angle>}
                    "tabletop": {<attr>: <value>}
                }
        """
        state = {"objects": [], "robot": {}, "tabletop": {}}

        # Get object states.
        for oid, odict in self.oid2attr.items():
            position, orientation = self.p.getBasePositionAndOrientation(oid)
            odict["oid"] = oid
            odict["position"] = list(position)
            odict["orientation"] = list(orientation)
            state["objects"].append(odict)

        # Get robot state.
        if self.robot_id is not None:
            for joint_idx in range(self.p.getNumJoints(self.robot_id)):
                joint_name = self.p.getJointInfo(self.robot_id, joint_idx)[
                    1
                ].decode("utf-8")
                joint_angle = self.p.getJointState(
                    bodyUniqueId=self.robot_id, jointIndex=joint_idx
                )[0]
                state["robot"][joint_name] = joint_angle

        # Get tabletop state.
        # position, _ = self.p.getBasePositionAndOrientation(oid)
        # state["tabletop"]["position"] = list(position)
        return copy.deepcopy(state)
