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
        mass_bounds: Optional[Tuple[float]] = None,
        mu_bounds: Optional[Tuple[float]] = None,
    ):
        """
        Args:
            n_objs_bounds: 2-tuple of min (inclusive) and max (exclusive) bounds 
                for the number of objects per scene.
            obj_dist_thresh: The minimum threshold distance between different 
                objects in the scene to enforced.
            max_retries: The maximum number of times to retry generating 
                values until certain criteria is met.
            shapes: A list of shapes to sample from.
            sizes: A list of sizes to sample from.
            # colors: A list of colors to sample from.
            radius_bounds: 2-tuple of the min and max bounds for shape radius.
                Note that box radius is downscaled by 0.8.
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

        self.mass_bounds = mass_bounds
        self.mu_bounds = mu_bounds

        self.obj_dist_thresh = obj_dist_thresh
        self.max_retries = max_retries
        self.position_mode = position_mode

    def generate_tabletop_objects(
        self, existing_odicts: Optional[List[Dict]] = None
    ) -> List[DashObject]:
        """Generates a single random scene.

        Note that box radius is downscaled by 0.8.
        
        Args:
            existing_odicts: A list of existing objects to include in the
                closeness tests so that new objects are not too close to the
                existing objects. Expected format: 
                [
                    {
                        "position": [x, y, z],
                        ...
                    },
                    ...
                ]

        Returns:
            odicts: A list of newly generated objects. Format:
            [
                {
                    "shape": <shape>,
                    "color": <color>,
                    "radius": <radius>,
                    "height": <height>,
                    "position": <position>,
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                }
            ]
        """
        if existing_odicts is None:
            existing_odicts = []

        # Randomly select the number of objects to generate.
        n_objects = self.generate_n_objects()

        odicts, n_tries = [], 0
        while len(odicts) < n_objects and n_tries < self.max_retries:
            odict = self.generate_object()

            # Check if generated object is too close to others.
            if self.any_close(
                src_odict=odict, other_odicts=odicts + existing_odicts
            ):
                n_tries += 1
            else:
                odicts.append(odict)
        return odicts

    def generate_object(self) -> DashObject:
        """Generates a random DashObject.

        Note that box radius is downscaled by 0.8.
        
        Returns:
            odict: The randomly generated object, in the format:
                {
                    "shape": <shape>,
                    "color": <color>,
                    "radius": <radius>,
                    "height": <height>,
                    "position": <position>,
                    "orientation": [0.0, 0.0, 0.0, 1.0]
                }

        """
        shape = random.choice(self.shapes)
        radius, height = self.generate_random_size(shape=shape)
        color = random.choice(self.colors)
        position = self.generate_random_xyz(
            self.x_bounds, self.y_bounds, self.z_bounds
        )
        if self.position_mode == "com":
            position[2] += height / 2
        elif self.position_mode == "base":
            pass
        else:
            raise ValueError(f"Invalid position mode: {self.position_mode}")

        odict = {
            "shape": shape,
            "color": color,
            "radius": radius,
            "height": height,
            "position": position,
            "orientation": [0.0, 0.0, 0.0, 1.0],
        }
        if self.mass_bounds is not None:
            odict["mass"] = self.uniform_sample(
                low=self.mass_bounds[0], high=self.mass_bounds[1]
            )
        if self.mu_bounds is not None:
            odict["mu"] = self.uniform_sample(
                low=self.mu_bounds[0], high=self.mu_bounds[1]
            )

        return odict

    def generate_n_objects(self) -> int:
        # `self.n_objs_bounds[1]` is exclusive while `random.randint` is
        # inclusive, so that's why we subtract one from the max.
        min_objs, max_objs = self.n_objs_bounds
        if min_objs == max_objs:
            n_objects = min_objs
        else:
            n_objects = random.randint(min_objs, max_objs - 1)
        return n_objects

    def generate_random_size(self, shape: str) -> Tuple[float, float]:
        """Generates a random radius and height. Note that box radius is 
        downscaled by 0.8.

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

        if shape == "box":
            radius *= 0.8
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

    def any_close(self, src_odict: Dict, other_odicts: List[Dict]) -> bool:
        """Checks if the source object is close to any of the other objects.

        Args:
            src_odict: The source object dictionary, with the format:
                {
                    "position": [x, y, z],
                    ...
                }
            other_odicts: The other object dictionaries, with the format:
                [
                    {
                        "position": [x, y, z],
                        ...
                    },
                    ...
                ]
        
        Returns:
            is_close: Whether the source object is close to any of the other 
                objects in xy space.
        """
        close_arr = [
            self.is_close(
                ax=src_odict["position"][0],
                ay=src_odict["position"][1],
                bx=other_odict["position"][0],
                by=other_odict["position"][1],
            )
            for other_odict in other_odicts
        ]
        is_close = any(close_arr)
        return is_close

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
