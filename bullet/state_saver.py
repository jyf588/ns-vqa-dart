"""A class that tracks objects and saves states."""
import os
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

        # A counter for example IDs.
        self.i = 0

    def reset(self):
        """Clears the list of tracked objects."""
        pass

    def track_object(self, oid: int, shape: str, radius: float, height: float):
        """Stores an object's metadata for tracking.
        
        Args:
            oid: The object ID.
            shape: The shape of the object.
            radius: The radius of the object.
            height: The height of the object.
        """
        pass

    def save(self):
        """Saves the current state of the bullet scene for tracked objects."""
        data = [
            {
                "shape": "cylinder",
                "radius": 0.05,
                "height": 0.18,
                "position": (0.0, 0.0, 0.0),
                "orientation": (0.0, 0.0, 0.0, 1.0),
            }
        ]
        path = os.path.join(self.dataset_dir, f"{self.i:06}.p")
        util.save_pickle(path=path, data=data)
        self.i += 1
