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

        self.objects = []

    def reset(self):
        """Resets the list of stored objects."""
        self.objects = []

    def add_object(self, o: DashObject):
        """Adds a single scene object.

        Args:
            o: A DashObject.
        """
        self.objects.append(o)

    # def add_object(
    #     self,
    #     oid: int,
    #     shape: str,
    #     color: str,
    #     radius: float,
    #     height: float,
    #     position: List[float],
    #     orientation: List[float],
    # ):
    #     """Adds a single scene object.

    #     Args:
    #         oid: PyBullet object ID.
    #         shape: The shape of the object.
    #         color: The color of the object.
    #         radius: The radius of the object.
    #         height: The height of the object.
    #         position: The xyz position of the center of the object's base, in
    #             world coordinate frame.
    #         orientation: The orientation of the object, expressed as a
    #             [x, y, z, w] quaternion, in world coordinate frame.
    #     """
    #     o = DashObject(
    #         shape=shape,
    #         color=color,
    #         radius=radius,
    #         height=height,
    #         position=position,
    #         orientation=orientation,
    #         oid=oid,
    #     )
    #     self.objects.append(o)

    def save_and_reset(self):
        """Saves the current scene example, and resets the stored information.
        """
        self.dataset.save_example(objects=self.objects, camera=self.camera)
        self.reset()
