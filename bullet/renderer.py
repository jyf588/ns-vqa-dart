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

from bullet.camera import BulletCamera
from bullet.dash_object import DashObject, DashTable


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

    # def render_table(self, o: DashTable) -> int:
    #     """Renders a DashTable.

    #     Returns:
    #         oid: The ID of the table.
    #     """
    #     self.load_urdf(
    #         shape=o.shape, position=o.position, orientation=o.orientation
    #     )
    #     oid = self.p.loadURDF(
    #         fileName=self.construct_object_path(obj_name=o.shape),
    #         basePosition=o.position,
    #         baseOrientation=o.orientation,
    #     )
    #     self.color_object(oid=oid, color=o.color)
    #     return oid

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
        rgba_color = COLOR2RGBA[o.color]

        if o.shape in PRIMITIVE2GEOM:
            oid = self.load_primitive(
                shape=o.shape,
                position=o.position,
                r=o.radius,
                h=o.height,
                rgba_color=rgba_color,
                check_sizes=check_sizes,
            )
        else:
            oid = self.load_nonprimitive(
                shape=o.shape, position=o.position, rgba_color=rgba_color
            )
        return oid

    def load_primitive(
        self,
        shape: str,
        position: List[float],
        r: float,
        h: Optional[float] = None,
        rgba_color: List[float] = None,
        check_sizes: Optional[bool] = True,
    ) -> Optional[int]:
        """Creates a primitive object.

        Args:
            shape: The name of the shape to generate.
            com_position: The (x, y, z) position of the COM, in world coordinates.
            r: The radius of the object.
            h: The height of the object. This should be 2*r for sphere.
            rgba_color: The color of the object.
            check_sizes: Whether to check that the sizes are valid for various
                shapes.

        Returns:
            If the object creation is successful:
                oid: The object ID.
            If the dimensions of the object are invalid:
                None
        """
        if check_sizes and shape == "sphere":
            assert h == 2 * r

        geom = PRIMITIVE2GEOM[shape]
        half_extents = [r, r, h / 2]

        # Update the position to be defined for the COM, since that's what
        # PyBullet expects.
        position[2] = h / 2

        # Create the shape.
        try:
            visualShapeId = self.p.createVisualShape(
                shapeType=geom,
                radius=r,
                halfExtents=half_extents,
                length=h,
                rgbaColor=rgba_color,
            )
        # This can happen when trying to render predictions that are physically
        # impossible.
        except pybullet.error as e:
            # print(e)
            # print(
            #     f"Invalid object. Shape: {shape}\tRadius: {r}\tHeight: {h}\tHalf_extents: {half_extents}"
            # )
            return None

        collisionShapeId = self.p.createCollisionShape(
            shapeType=geom, radius=r, halfExtents=half_extents, height=h
        )
        oid = self.p.createMultiBody(
            baseMass=3.5,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=position,  # Pybullet expects COM.
        )
        return oid

    def load_nonprimitive(
        self, shape: str, position: List[float], rgba_color: List[float]
    ) -> int:
        """Loads a nonprimitive object.

        Args:
            shape: The object shape.
            position: The position to set the origin of the object.
            rgba_color: The RGBA color of the object.
        
        Returns:
            oid: The object ID.
        """
        path = self.construct_object_path(obj_name=shape)
        if shape in URDF_SHAPES:
            oid = self.load_urdf(
                path=path, position=position, rgba_color=rgba_color
            )
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
                rgba_color=rgba_color,
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
        rgba_color: List[float],
    ):
        """Loads a mesh object.

        Args:
            path: The path to the object file.
            position: The xyz position of the object's origin.
            # orientation: The orientation of the object.
            rgba_color: The color of the object.

        Returns:
            oid: The object ID.
        """
        visualShapeId = self.p.createVisualShape(
            shapeType=self.p.GEOM_MESH,
            fileName=path,
            rgbaColor=rgba_color,
            meshScale=scale,
            visualFrameOrientation=orientation,
        )
        collisionShapeId = self.p.createCollisionShape(
            shapeType=self.p.GEOM_MESH, fileName=path, meshScale=scale
        )

        oid = self.p.createMultiBody(
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=position,
        )
        return oid

    def load_urdf(
        self, path: str, position: List[float], rgba_color: List[float]
    ):
        """Loads a urdf object.

        Args:
            path: The path of the urdf file.
            position: The position to set the origin of the object.
            rgba_color: The RGBA color of the object.
        
        Returns:
            oid: The object ID.
        """
        oid = self.p.loadURDF(path, basePosition=position)
        self.p.changeVisualShape(
            objectUniqueId=oid, linkIndex=-1, rgbaColor=rgba_color
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
