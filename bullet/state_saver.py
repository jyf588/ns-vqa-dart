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

        self.oid2odict = {}

        # A counter for example IDs.
        self.i = 0

    def reset(self):
        """Clears the list of tracked objects."""
        self.oid2odict = {}

    def track_object(self, oid: int, shape: str, radius: float, height: float):
        """Stores an object's metadata for tracking.
        
        Args:
            oid: The object ID.
            shape: The shape of the object.
            radius: The radius of the object.
            height: The height of the object.
        """
        self.oid2odict[oid] = {
            "shape": shape,
            "radius": radius,
            "height": height,
        }

    def save(self):
        """Saves the current state of the bullet scene for tracked objects."""
        self.update_state()
        data = list(self.oid2odict.values())
        # data = [
        #     {
        #         "shape": "cylinder",
        #         "radius": 0.05,
        #         "height": 0.18,
        #         "position": [0.0, 0.0, 0.0],
        #         "orientation": [0.0, 0.0, 0.0, 1.0],
        #     },
        #     {
        #         "shape": "box",
        #         "radius": 0.04,
        #         "height": 0.16,
        #         "position": [0.1, 0.1, 0.0],
        #         "orientation": [0.0, 0.0, 0.0, 1.0],
        #     },
        # ]
        path = os.path.join(self.dataset_dir, f"{self.i:06}.p")
        util.save_pickle(path=path, data=data)
        self.i += 1

    def update_state(self):
        """Update the state (position and orientation) of objects we are
        tracking.

        Note that position is based on the COM of the object.
        """
        for oid, odict in self.oid2odict.items():
            position, orientation = p.getBasePositionAndOrientation(oid)
            odict["position"] = list(position)
            odict["orientation"] = list(orientation)
            self.oid2odict[oid] = odict
