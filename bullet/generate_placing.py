import pybullet_utils.bullet_client as bc
from typing import *

from .camera import BulletCamera
from .dash_dataset import DashDataset
from .dash_object import DashObject


class PlacingDatasetGenerator:
    """Generates a vision dataset containing examples of placing. To be used
    in placing environments.
    """

    def __init__(
        self,
        p: bc.BulletClient,
        dataset_dir: str,
        camera_position: Optional[List[float]] = [
            -0.1916501582752709,
            0.03197646764976494,
            0.4177423103840716,
        ],
        camera_rotation: Optional[List[float]] = [0.0, 50.0, 0.0],
        camera_offset: Optional[List[float]] = [0.0, 0.0, 0.0],
    ):
        """
        Args:
            p: The bullet client to use for generating the dataset.
            dataset_dir: The directory to save the dataset in.
            camera_position: The position of the camera.
            camera_rotation: The roll, pitch, and yaw of the camera (degrees).
            camera_offset: The amount to offset the camera position compared to
                the camera position that the vision module was trained on.
        
        Attributes:
            p: The bullet client to use for generating the dataset.
            objects: A running list of DashObjects currently in an example 
                scene.
        """
        self.p = p
        self.dataset = DashDataset(dataset_dir=dataset_dir)
        self.camera = BulletCamera(
            p=p,
            position=camera_position,
            rotation=camera_rotation,
            offset=camera_offset,
        )

        self.oid2object: Dict[int, DashObject] = {}

    def reset(self):
        """Resets the list of stored objects."""
        self.oid2object = {}

    def track_object(self, o: DashObject):
        """Adds an object to track for dataset saving.

        Args:
            o: A DashObject.
        """
        self.oid2object[o.oid] = o

    def generate_example(self):
        """Generates a dataset example from the current bullet state."""
        self.update_state()
        objects = list(self.oid2object.values())
        self.dataset.save_example(objects=objects, camera=self.camera)

    def update_state(self):
        """Updates the current state. Currently, only the position and 
        orientation of objects are updated in the function.
        """
        for oid in self.oid2object.keys():
            o = self.oid2object[oid]
            o.position, o.orientation = self.p.getBasePositionAndOrientation(
                oid
            )
            self.oid2object[oid] = o
