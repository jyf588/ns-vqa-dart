import numpy as np
import random
import time
from tqdm import tqdm
from typing import List, Tuple

from bullet.camera import BulletCamera
from bullet.dash_dataset import DashDataset
from bullet.dash_object import DashObject, DashTable, DashRobot
from bullet.renderer import BulletRenderer
import bullet.util


class RandomSceneGenerator:
    def __init__(
        self,
        seed: int,
        render_mode: str,
        egocentric: bool,
        dataset_dir: str,
        roll_bounds: Tuple[float],
        tilt_bounds: Tuple[float],
        pan_bounds: Tuple[float],
        degrees: bool,
    ):
        """
        Note: All lower bounds are inclusive. All upper bounds are exclusive.

        Args:
            seed: The seed for numpy random number generator.
            render_mode: The mode to render in, i.e., CPU-based TinyRenderer 
                ("direct") or GPU-based OpenGL ("gui").
            egocentric: Whether the camera viewpoint is egocentric.
            dataset_dir: The directory to store the data to.
            n_objs_bounds: 2-tuple of min and max bounds for the number of 
                objects per scene.
            obj_dist_thresh: The minimum threshold distance between different 
                objects in the scene to enforced.
            max_retries: The maximum number of times to retry generating 
                values until certain criteria is met.
            shapes: A list of shapes to sample from.
            sizes: A list of sizes to sample from.
            colors: A list of colors to sample from.
            radius_bounds: 2-tuple of the min and max bounds for shape radius.
            height_bounds: 2-tuple of the min and max bounds for shape height.
            x_bounds: 2-tuple of the min and max bounds for the x position.
            y_bounds: 2-tuple of the min and max bounds for the y position.
            z_bounds: 2-tuple of the min and max bounds for the z position.
            roll_bounds: 2-tuple of the min and max bounds for the camera roll.
            tilt_bounds: 2-tuple of the min and max bounds for the camera tilt.
            pan_bounds: 2-tuple of the min and max bounds for the camera pan.
            degrees: Whether the roll, tilt, and pan is expressed in terms of
                degrees (true) or radians (false).
        
        Attributes:
            p: The bullet client to be used for rendering.
            renderer: The BulletRenderer for rendering.
            robot: The DashRobot for egocentric views.
            camera: A BulletCamera for generating images in non-egocentric mode.
        """

        self.obj_dist_thresh = obj_dist_thresh
        self.max_retries = max_retries

        # Initialize the Bullet client and renderer.
        self.p = bullet.util.create_bullet_client(mode=render_mode)
        self.renderer = BulletRenderer(p=self.p)

        # Define the camera as default or egocentric.
        self.egocentric = egocentric
        if self.egocentric:
            self.robot = DashRobot(p=self.p)
        else:
            self.camera = BulletCamera(init_type="default")

        # Initialize the dataset generator.
        self.dataset = DashDataset(dataset_dir=dataset_dir)

        # Initialize the tabletop which is constant throughout the scenes.
        self.renderer.render_object(DashTable())

        # Define randomizable parameters.
        self.n_objs_bounds = n_objs_bounds
        self.shapes = shapes
        self.colors = colors
        self.radius_bounds = radius_bounds
        self.height_bounds = height_bounds
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds
        self.roll_bounds = roll_bounds
        self.tilt_bounds = tilt_bounds
        self.pan_bounds = pan_bounds
        self.degrees = degrees

    def generate_scenes(self, n: int):
        """Generates a specified number of randomized scenes.

        Args:
            n: The number of scenes to generate.
        """
        for _ in tqdm(range(n)):
            objects = self.generate_tabletop_objects(
                n_objs_bounds=self.n_objs_bounds
            )
            objects = self.renderer.render_objects(objects=objects)
            if self.egocentric:
                self.generate_and_set_new_camera_view()
            camera = self.robot.camera if self.egocentric else self.camera
            self.dataset.save_example(objects=objects, camera=camera)
            self.renderer.remove_objects(objects=objects)

    def generate_and_set_new_camera_view(self):
        """Generates and sets new camera viewpoint."""
        roll, tilt, pan = self.generate_random_xyz(
            self.roll_bounds, self.tilt_bounds, self.pan_bounds
        )
        self.robot.set_head_and_camera(
            roll=roll, tilt=tilt, pan=pan, degrees=self.degrees
        )

