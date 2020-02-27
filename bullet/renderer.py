"""
Definition of a BulletRenderer which renders various objects in a bullet scene.
"""

import math
import numpy as np
import os
import pybullet
import pybullet_utils.bullet_client as bc
from typing import *

from bullet.camera import BulletCamera
from bullet.dash_object import DashObject, DashTable


SHAPE2GEOM = {
    "box": pybullet.GEOM_BOX,
    "cylinder": pybullet.GEOM_CYLINDER,
    "sphere": pybullet.GEOM_SPHERE,
}

COLOR2RGBA = {
    "red": [0.8, 0.0, 0.0, 1.0],
    "grey": [0.4, 0.4, 0.4, 1.0],
    "yellow": [0.8, 0.8, 0.0, 1.0],
    "blue": [0.0, 0.0, 0.8, 1.0],
    "green": [0.0, 0.8, 0.0, 1.0],
}


class BulletRenderer:
    def __init__(self, p, urdf_dir="bullet/assets"):
        # self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.p = p
        self.urdf_dir = urdf_dir

    def render_table(self, o: DashTable) -> int:
        """Renders a DashTable.

        Returns:
            oid: The ID of the table.
        """
        oid = self.p.loadURDF(
            fileName=self.construct_urdf_path(o),
            basePosition=o.position,
            baseOrientation=o.orientation,
        )
        self.color_object(oid=oid, color=o.color)
        return oid

    def render_object(
        self, o: DashObject, check_sizes: Optional[bool] = True
    ) -> Optional[int]:
        """Renders a DashObject.

        Args:
            o: A DashObject.
            check_sizes: Whether to check sizes of the object.
        
        Returns:
            oid: The object ID.
        """
        if o.shape == "lego":
            oid = self.p.loadURDF(
                os.path.join(self.urdf_dir, "lego.urdf"),
                basePosition=o.position,
            )
        elif o.shape == "cup":
            oid = self.p.loadURDF(
                os.path.join(self.urdf_dir, "cup/cup_small.urdf"),
                basePosition=o.position,
            )
        else:
            oid = self.generate_primitive_shape(
                shape=o.shape,
                position=o.position,
                r=o.radius,
                h=o.height,
                check_sizes=check_sizes,
            )
        if oid is not None and o.color is not None:
            self.color_object(oid=oid, color=o.color)
        return oid

    def color_object(self, oid: int, color: str):
        """Colors an object.

        Args:
            oid: The ID of the object.
            color: The color to color the object.
        """
        self.p.changeVisualShape(
            objectUniqueId=oid, linkIndex=-1, rgbaColor=COLOR2RGBA[color]
        )

    def generate_primitive_shape(
        self,
        shape: str,
        position: List[float],
        r: float,
        h: Optional[float] = None,
        check_sizes: Optional[bool] = True,
    ) -> Optional[int]:
        """Creates a primitive object.

        Args:
            shape: The name of the shape to generate.
            com_position: The (x, y, z) position of the COM, in world coordinates.
            r: The radius of the object.
            h: The height of the object. This should be 2*r for sphere.
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

        geom = SHAPE2GEOM[shape]
        half_extents = [r, r, h / 2]

        # Update the position to be defined for the COM, since that's what
        # PyBullet expects.
        position[2] = h / 2

        # Create the shape.
        try:
            visualShapeId = self.p.createVisualShape(
                shapeType=geom, radius=r, halfExtents=half_extents, length=h
            )
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

    def construct_urdf_path(self, o: DashTable) -> str:
        """Constructs the URDF path based on object attributes.
        
        Args:
            o: A DashObject.
        
        Returns:
            urdf_path: The path to the urdf file matching the attributes of the
                DashObject.
        """
        urdf_path = os.path.join(self.urdf_dir, f"{o.shape}.urdf")
        return urdf_path
