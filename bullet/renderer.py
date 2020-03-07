"""
Definition of a BulletRenderer which renders various objects in a bullet scene.
"""

import copy
import math
import numpy as np
import os
import pybullet
import pybullet_utils.bullet_client as bc
from typing import *

from .camera import BulletCamera
from .dash_object import DashObject, DashTable


PRIMITIVE2GEOM = {
    "box": pybullet.GEOM_BOX,
    "cylinder": pybullet.GEOM_CYLINDER,
    "sphere": pybullet.GEOM_SPHERE,
}

SHAPE2PATH = {
    "table": "table.urdf",
    "lego": "lego.urdf",
    # "cup": "cup/cup_small.urdf",
    "cup": "cup/Cup/cup_vhacd.obj",
    "soda_can": "soda_can.obj",
    "can": "can.obj",
}

URDF_SHAPES = ["table", "lego"]
MESH_SHAPES = ["soda_can", "cup", "can"]

MESH_MEASUREMENTS = {
    "soda_can": {
        "scale": [0.22, 0.24, 0.22],
        "origin": [0.0, 0.0, 0.115],
        "orientation": [
            math.cos(math.radians(90 / 2)),
            0,
            0,
            math.sin(math.radians(90 / 2)),
        ],
    },
    "can": {
        "scale": [0.22, 0.24, 0.22],
        "origin": [0.0, 0.0, 0.115],
        "orientation": [
            math.cos(math.radians(90 / 2)),
            0,
            0,
            math.sin(math.radians(90 / 2)),
        ],
    },
}

COLOR2RGBA = {
    "red": [0.8, 0.0, 0.0, 1.0],
    "grey": [0.4, 0.4, 0.4, 1.0],
    "yellow": [0.8, 0.8, 0.0, 1.0],
    "blue": [0.0, 0.0, 0.8, 1.0],
    "green": [0.0, 0.8, 0.0, 1.0],
}


class BulletRenderer:
    def __init__(self, p, assets_dir="bullet/assets"):
        self.p = p
        self.assets_dir = assets_dir

    def render_object(
        self, o: DashObject, check_sizes: Optional[bool] = True
    ) -> Optional[int]:
        """Renders a DashObject.

        Args:
            o: A DashObject.
            check_sizes: Whether to check sizes of the object, e.g. that the
                height of a sphere should be 2*r.
        
        Returns:
            oid: The object ID.
        """
        # Make a deep copy of the object because downstream functions might
        # modify the object for rendering purposes.
        o = copy.deepcopy(o)

        if o.shape in PRIMITIVE2GEOM:
            oid = self.create_primitive(
                geom=PRIMITIVE2GEOM[o.shape],
                position=o.position,
                r=o.radius,
                h=o.height,
                color=color,
                check_sizes=check_sizes,
            )
        else:
            oid = self.load_nonprimitive(
                shape=o.shape, position=o.position, color=color
            )
        return oid

    def create_primitive(
        self,
        geom: Any,
        base_position: List[float],
        orientation: List[float],
        r: float,
        h: Optional[float] = None,
        color: str = None,
        check_sizes: Optional[bool] = True,
        base_mass: Optional[float] = 3.5,
    ) -> Optional[int]:
        """Creates a primitive object.

        Args:
            geom: The bullet GEOM_{shape} enum.
            base_position: The (x, y, z) base position of the object, in 
                world coordinates.
            orientation: The orientation of the object, in [x, y, z, w] 
                quaternion format.
            r: The radius of the object.
            h: The height of the object. This should be 2*r for sphere.
            color: The color of the object.
            check_sizes: Whether to check that the sizes are valid for various
                shapes.
            base_mass: The mass of the object.

        Returns:
            If the object creation is successful:
                oid: The object ID.
            If creating the visual shape failed:
                None
        
        Raises:
            pybullet.error: If creating the visual shape failed (e.g., with
                invalid dimensions). Only raised if `check_sizes` is True.
        """
        if check_sizes and geom == pybullet.GEOM_SPHERE:
            assert h == 2 * r

        half_extents = [r, r, h / 2]

        # Update the position to be defined for the COM, since that's what
        # PyBullet expects. So we offset the z position by half of the height.
        com_position = base_position.copy()
        com_position[2] += h / 2

        # Create the shape.
        try:
            visual_shape_id = self.p.createVisualShape(
                shapeType=geom,
                radius=r,
                halfExtents=half_extents,
                length=h,
                rgbaColor=COLOR2RGBA[color],
            )
        # Errors can occur when trying to render predictions that are
        # physically impossible.
        except pybullet.error as e:
            print(
                f"Attempted rendering of invalid object. "
                f"Shape: {shape}\t"
                f"Radius: {r}\t"
                f"Height: {h}\t"
                f"Half_extents: {half_extents}"
            )

            if check_sizes:
                raise (e)

            # Don't raise the exception, in cases when we are rendering e.g.
            # predictions (which we expect to be invalid at times).
            else:
                return None

        collision_shape_id = self.p.createCollisionShape(
            shapeType=geom, radius=r, halfExtents=half_extents, height=h
        )
        oid = self.p.createMultiBody(
            baseMass=base_mass,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=com_position,  # Pybullet expects COM.
            baseOrientation=orientation,
        )
        return oid

    def load_nonprimitive(
        self, shape: str, position: List[float], color: str
    ) -> int:
        """Loads a nonprimitive object.

        Args:
            shape: The object shape.
            position: The position to set the origin of the object.
            color: The color of the object.
        
        Returns:
            oid: The object ID.
        """
        path = self.construct_object_path(obj_name=shape)
        if shape in URDF_SHAPES:
            oid = self.load_urdf(path=path, position=position, color=color)
        elif shape in MESH_SHAPES:
            mesh_measurements = MESH_MEASUREMENTS[shape]

            scale = mesh_measurements["scale"]
            origin = mesh_measurements["origin"]
            position = list(np.array(position) + np.array(origin))
            orientation = mesh_measurements["orientation"]

            oid = self.load_mesh(
                path=path,
                scale=scale,
                position=position,
                orientation=orientation,
                color=color,
            )
        else:
            raise ValueError(
                f"Invalid shape: {shape}. Shape must be a primitive, URDF, or MESH."
            )
        return oid

    def load_mesh(
        self,
        path: str,
        scale: List[float],
        position: List[float],
        orientation: List[float],
        color: str,
    ):
        """Loads a mesh object.

        Args:
            path: The path to the object file.
            position: The xyz position of the object's origin.
            # orientation: The orientation of the object.
            color: The color of the object.

        Returns:
            oid: The object ID.
        """
        visual_shape_id = self.p.createVisualShape(
            shapeType=self.p.GEOM_MESH,
            fileName=path,
            rgbaColor=COLOR2RGBA[color],
            meshScale=scale,
            visualFrameOrientation=orientation,
        )
        collisionShapeId = self.p.createCollisionShape(
            shapeType=self.p.GEOM_MESH, fileName=path, meshScale=scale
        )

        oid = self.p.createMultiBody(
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
        )
        return oid

    def load_urdf(self, path: str, position: List[float], color: str):
        """Loads a urdf object.

        Args:
            path: The path of the urdf file.
            position: The position to set the origin of the object.
            color: The color of the object.
        
        Returns:
            oid: The object ID.
        """
        oid = self.p.loadURDF(path, basePosition=position)
        self.p.changeVisualShape(
            objectUniqueId=oid, linkIndex=-1, rgbaColor=COLOR2RGBA[color]
        )
        return oid

    def construct_object_path(self, obj_name: str) -> str:
        """Constructs the URDF path based on object attributes.
        
        Args:
            obj_name: The name of the object.
        
        Returns:
            path: The path to the file for the object.
        """
        path = os.path.join(self.assets_dir, SHAPE2PATH[obj_name])
        return path
