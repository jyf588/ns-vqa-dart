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
        n_objs_bounds: Tuple[int],
        obj_dist_thresh: float,
        max_retries: int,
        shapes: List[str],
        sizes: List[str],
        colors: List[str],
        x_bounds: Tuple[float],
        y_bounds: Tuple[float],
        z_bounds: Tuple[float],
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
        # Set the seed for numpy's random number generator.
        np.random.seed(seed)
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
            self.camera = BulletCamera(p=self.p)

        # Initialize the dataset generator.
        self.dataset = DashDataset(dataset_dir=dataset_dir)

        # Initialize the tabletop which is constant throughout the scenes.
        self.renderer.render_object(DashTable())

        # Define randomizable parameters.
        self.n_objs_bounds = n_objs_bounds
        self.shapes = shapes
        self.sizes = sizes
        self.colors = colors
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
            objects = self.generate_scene()
            rgb, mask = self.generate_image()
            self.dataset.save_example(
                objects=objects, camera=self.robot.camera, rgb=rgb, mask=mask
            )
            self.remove_objects(objects=objects)

    def remove_objects(self, objects: List[DashObject]):
        """Removes objects from the scene.
        
        Args:
            objects: A list of DashObjects to be removed from the scene.
        """
        for o in objects:
            assert o.oid is not None
            self.p.removeBody(o.oid)

    def generate_scene(self) -> List[DashObject]:
        """Generates a single random scene."""
        # self.p.resetSimulation()

        # Randomly select the number of objects to generate.
        # `self.n_objs_bounds[1]` is exclusive while `random.randint` is
        # inclusive, so that's why we subtract one from the max.
        min_objs, max_objs = self.n_objs_bounds
        if min_objs == max_objs:
            n_objects = min_objs
        else:
            n_objects = random.randint(min_objs, max_objs - 1)

        objects = []
        n_tries = 0
        while len(objects) < n_objects and n_tries < self.max_retries:
            o: DashObject = self.generate_object()

            # Check if generated object is too close to others.
            close_arr = [
                self.is_close(
                    ax=o.position[0],
                    ay=o.position[1],
                    bx=other.position[0],
                    by=other.position[1],
                )
                for other in objects
            ]
            if any(close_arr):
                n_tries += 1
            else:
                oid = self.renderer.render_object(o, fix_base=True)
                o.oid = oid
                objects.append(o)
        return objects

    def generate_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Takes a snapshot of the current scene, where viewpoint is randomly
        chosen.

        Returns:
            rgb: The RGB image of the scene.
            mask: The mask of the scene, where values represent the object ID
                that is present at each pixel.
        """
        roll, tilt, pan = self.generate_xyz(
            self.roll_bounds, self.tilt_bounds, self.pan_bounds
        )
        if self.egocentric:
            self.robot.set_head_and_camera(
                roll=roll, tilt=tilt, pan=pan, degrees=self.degrees
            )
            rgb, mask = self.robot.camera.get_rgb_and_mask(p=self.p)
        else:
            raise NotImplementedError
        return rgb, mask

    def generate_object(self) -> DashObject:
        """Generates a random DashObject.
        
        Returns:
            o: The randomly generated DashObject.
        """

        o = DashObject(
            shape=random.choice(self.shapes),
            size=random.choice(self.sizes),
            color=random.choice(self.colors),
            position=self.generate_xyz(
                self.x_bounds, self.y_bounds, self.z_bounds
            ),
            orientation=[0.0, 0.0, 0.0, 1.0],
        )
        return o

    def generate_xyz(self, x_bounds, y_bounds, z_bounds) -> List[int]:
        """Generates a random xyz based on axis bounds.

        Returns:
            position: The randomly generated xyz position.
        """
        xyz = []
        for (axis_min, axis_max) in [x_bounds, y_bounds, z_bounds]:
            if axis_min == axis_max:  # No randomization.
                axis_value = axis_min
            elif axis_min < axis_max:
                axis_value = np.random.uniform(axis_min, axis_max)
            else:
                raise ValueError(
                    f"Invalid axis bounds: ({axis_min}, {axis_max})"
                )
            xyz.append(axis_value)
        return xyz

    def is_close(self, ax: float, ay: float, bx: float, by: float) -> bool:
        """Checks whether two (x, y) points are within a certain threshold 
        distance of each other.

        Args:
            ax: The x position of the first point.
            ay: The y position of the first point.
            bx: The x position of the second point.
            by: The y position of the second point.
        
        Returns:
            Whether the distance between the two points is less than or equal
                to the threshold distance.
        """
        return (ax - bx) ** 2 + (ay - by) ** 2 <= self.obj_dist_thresh ** 2
