import numpy as np
import random
from typing import *

from ns_vqa_dart.bullet.dash_object import DashObject
import ns_vqa_dart.bullet.renderer as bullet_renderer


class RandomObjectsGenerator:
    def __init__(
        self,
        seed: int,
        n_objs_bounds: Tuple[int],
        obj_dist_thresh: float,
        max_retries: int,
        shapes: List[str],
        colors: List[str],
        radius_bounds: Tuple[float],
        height_bounds: Tuple[float],
        x_bounds: Tuple[float],
        y_bounds: Tuple[float],
        z_bounds: Tuple[float],
        position_mode: str,
    ):
        """
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
        z_bounds: 2-tuple of the min and max bounds for the z position. Note 
            that this represents the base Z position. The final generated Z
            position will be offset by +H/2 if the `position_mode` is set to
            "com".
        position_mode: Whether the position represents the base or the COM.
        """
        # Set the seed for random number generators.
        np.random.seed(seed)
        random.seed(seed)

        # Define randomizable parameters.
        self.n_objs_bounds = n_objs_bounds
        self.shapes = shapes
        self.colors = colors
        self.radius_bounds = radius_bounds
        self.height_bounds = height_bounds
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.z_bounds = z_bounds

        self.obj_dist_thresh = obj_dist_thresh
        self.max_retries = max_retries
        self.position_mode = position_mode

    def generate_tabletop_objects(
        self, existing_objects: Optional[List[DashObject]] = None
    ) -> List[DashObject]:
        """Generates a single random scene.
        
        Args:
            existing_objects: A list of existing objects to consider when 
                checking closeness.

        Returns:
            objects: A list of DashObjects generated in the scene.
        """
        if existing_objects is None:
            existing_objects = []

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
                for other in objects + existing_objects
            ]
            if any(close_arr):
                n_tries += 1
            else:
                objects.append(o)
        return objects

    def generate_object(self) -> DashObject:
        """Generates a random DashObject.
        
        Returns:
            o: The randomly generated DashObject.
        """
        shape = random.choice(self.shapes)
        radius, height = self.generate_random_size(shape=shape)
        color = generate_random_color()
        position = self.generate_random_xyz(
            self.x_bounds, self.y_bounds, self.z_bounds
        )
        if self.position_mode == "com":
            position[2] += height / 2
        elif self.position_mode == "base":
            pass
        else:
            raise ValueError(f"Invalid position mode: {self.position_mode}")

        o = DashObject(
            shape=shape,
            radius=radius,
            height=height,
            color=color,
            position=position,
            orientation=[0.0, 0.0, 0.0, 1.0],
        )
        return o

    def generate_random_size(self, shape: str) -> Tuple[float, float]:
        """Generates a random radius and height.

        Args:
            shape: The shape we are generating a size for.

        Returns:
            radius: The radius of the object.
            height: The height of the object. This is 2*r for sphere.
        """
        min_r, max_r = self.radius_bounds
        min_h, max_h = self.height_bounds
        radius = self.uniform_sample(low=min_r, high=max_r)
        if shape == "sphere":
            height = radius * 2
        else:
            height = self.uniform_sample(low=min_h, high=max_h)
        return radius, height

    def generate_random_xyz(self, x_bounds, y_bounds, z_bounds) -> List[int]:
        """Generates a random xyz based on axis bounds.

        Returns:
            xyz: The randomly generated xyz values.
        """
        xyz = []
        for (axis_min, axis_max) in [x_bounds, y_bounds, z_bounds]:
            axis_value = self.uniform_sample(low=axis_min, high=axis_max)
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

    def uniform_sample(self, low: float, high: float) -> float:
        if low == high:  # No randomization.
            value = low
        elif low < high:
            value = np.random.uniform(low=low, high=high)
        else:
            raise ValueError(
                f"Invalid bounds for uniform sample: ({low}, {high})"
            )
        return value


def generate_random_color():
    color = random.choice(bullet_renderer.OBJECT_COLORS)
    return color
