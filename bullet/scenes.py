import numpy as np
import random
import time
from tqdm import tqdm
from typing import List, Tuple

from bullet.camera import BulletCamera
from bullet.dash_object import DashObject, DashTable
from bullet.renderer import BulletRenderer
import bullet.util


class RandomSceneGenerator:
    def __init__(
        self,
        seed: int,
        render_mode: str,
        n_objs_bounds: Tuple[int],
        shapes: List[str],
        sizes: List[str],
        colors: List[str],
        x_bounds: Tuple[float],
        y_bounds: Tuple[float],
        z_bounds: Tuple[float],
    ):
        """
        Note: All lower bounds are inclusive. All upper bounds are exclusive.

        Args:
            seed: The seed for numpy random number generator.
            x_bounds: 2-tuple of the min and max bounds for the x position.
            y_bounds: 2-tuple of the min and max bounds for the y position.
            z_bounds: 2-tuple of the min and max bounds for the z position.
        """
        # Set the seed for numpy's random number generator.
        np.random.seed(seed)

        # Initialize the Bullet client, renderer, and camera.
        self.p = bullet.util.create_bullet_client(mode=render_mode)
        self.renderer = BulletRenderer(p=self.p)
        self.camera = BulletCamera(p=self.p)

        # Initialize the tabletop which is constant throughout the scenes.
        self.table = DashTable()

        # Define randomizable parameters.
        self.n_objs_bounds = n_objs_bounds
        self.shapes = shapes
        self.sizes = sizes
        self.colors = colors
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

    def generate_scenes(self, n: int):
        """Generates a specified number of randomized scenes.

        Args:
            n: The number of scenes to generate.
        """
        for _ in tqdm(range(n)):
            self.generate_scene()
            self.camera.get_rgb_and_mask()
            time.sleep(0.25)

    def generate_scene(self):
        """Generates a single random scene."""
        self.p.resetSimulation()
        self.renderer.render_object(self.table)

        # Randomly select the number of objects to generate.
        # `self.n_objs_bounds[1]` is exclusive while `random.randint` is
        # inclusive, so that's why we subtract one from the max.
        n_objects = random.randint(
            self.n_objs_bounds[0], self.n_objs_bounds[1] - 1
        )
        for _ in range(n_objects):
            o: DashObject = self.generate_object()
            self.renderer.render_object(o, fix_base=True)

    def generate_object(self) -> DashObject:
        """Generates a random DashObject.
        
        Returns:
            o: The randomly generated DashObject.
        """

        o = DashObject(
            shape=random.choice(self.shapes),
            size=random.choice(self.sizes),
            color=random.choice(self.colors),
            world_position=self.generate_position(),
            world_orientation=[0.0, 0.0, 0.0, 1.0],
        )
        return o

    def generate_position(self) -> List[int]:
        """Generates a random position based on axis bounds.

        Returns:
            position: The randomly generated xyz position.
        """
        position = []
        for (axis_min, axis_max) in [
            self.x_bounds,
            self.y_bounds,
            self.z_bounds,
        ]:
            if axis_min == axis_max:  # No randomization.
                axis_pos = axis_min
            elif axis_min < axis_max:
                axis_pos = np.random.uniform(axis_min, axis_max)
            else:
                raise ValueError(
                    f"Invalid axis bounds: ({axis_min}, {axis_max})"
                )
            position.append(axis_pos)
        return position
