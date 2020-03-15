"""
Definition of a BulletRenderer which renders various objects in a bullet scene.
"""

import copy
import math
import numpy as np
import os
import pybullet
import pybullet_utils.bullet_client as bc
import random
from typing import *

from ns_vqa_dart.bullet.camera import BulletCamera
import ns_vqa_dart.bullet.dash_object as dash_object
from ns_vqa_dart.bullet.dash_object import DashObject, DashTable, DashRobot


PRIMITIVE2GEOM = {
    "box": pybullet.GEOM_BOX,
    "cylinder": pybullet.GEOM_CYLINDER,
    "sphere": pybullet.GEOM_SPHERE,
}

SHAPE2PATH = {
    "tabletop": "tabletop.urdf",
    "lego": "lego.urdf",
    # "cup": "cup/cup_small.urdf",
    "cup": "cup/Cup/cup_vhacd.obj",
    "soda_can": "soda_can.obj",
    "can": "can.obj",
}

URDF_SHAPES = ["tabletop", "lego"]
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

OBJECT_COLORS = ["red", "yellow", "blue", "green"]
COLOR2RGBA = {
    "red": [0.8, 0.0, 0.0, 1.0],
    "grey": [0.4, 0.4, 0.4, 1.0],
    "yellow": [0.8, 0.8, 0.0, 1.0],
    "blue": [0.0, 0.0, 0.8, 1.0],
    "green": [0.0, 0.8, 0.0, 1.0],
}


class BulletRenderer:
    def __init__(
        self, p=None, assets_dir: Optional[str] = "my_pybullet_envs/assets"
    ):
        self.p = p
        self.assets_dir = assets_dir

    def set_bullet_client(self, p):
        self.p = p

    def remove_objects(self, ids: List[int]):
        """Removes objects from the scene.
        
        Args:
            objects: A list of DashObjects to be removed from the scene.
        """
        for id_to_del in ids:
            assert id_to_del is not None
            self.p.removeBody(id_to_del)

    def load_objects_from_state(self, ostates: List[Dict], position_mode: str):
        """Loads objects from object state.

        Args:
            ostates: A list of dictionaries representing object states.
            position_mode: Whether the position represents the base or the COM.

        Returns:
            oids: A list of object ids loaded, with order corresponding to 
                input ostates.
        """
        ostates = copy.deepcopy(ostates)

        objects = []
        for odict in ostates:
            odict["oid"], odict["img_id"] = None, None

            # Convert to DashObject.
            o = dash_object.from_json(odict)
            objects.append(o)

        # Render objects.
        objects = self.render_objects(
            objects=objects, position_mode=position_mode
        )
        oids = [o.oid for o in objects]
        return oids

    def render_objects(
        self,
        objects: List[DashObject],
        position_mode: str,
        base_mass: Optional[float] = 3.5,
        use_fixed_base: Optional[bool] = False,
        check_sizes: Optional[bool] = True,
    ) -> List[DashObject]:
        """Renders multiple Dash Objects.

        Args:
            objects: A list of DashObject's to render.
            position_mode: Whether the position represents the base or the COM.
            base_mass: The mass of the object.
            use_fixed_base: Whether to force the base of the loaded object to
                be static. Used for loading URDF objects.
            check_sizes: Whether to check sizes of the object, e.g. that the
                height of a sphere should be 2*r.

        Returns:
            objects_w_oid: The same list of input objects, but with the object
                IDs set.
        """
        objects_w_oid = []
        for o in objects:
            oid = self.render_object(
                o=o,
                position_mode=position_mode,
                base_mass=base_mass,
                use_fixed_base=use_fixed_base,
                check_sizes=check_sizes,
            )
            o.oid = oid
            objects_w_oid.append(o)
        return objects_w_oid

    def render_object(
        self,
        o: DashObject,
        position_mode: str,
        base_mass: Optional[float] = 3.5,
        use_fixed_base: Optional[bool] = False,
        check_sizes: Optional[bool] = True,
    ) -> Optional[int]:
        """Renders a DashObject.

        Args:
            o: A DashObject, with all attributes except for oid and img_id 
                filled.
            position_mode: Whether the position represents the base or the COM.
            base_mass: The mass of the object.
            use_fixed_base: Whether to force the base of the loaded object to
                be static. Used for loading URDF objects.
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
                position_mode=position_mode,
                position=o.position,
                orientation=o.orientation,
                r=o.radius,
                h=o.height,
                color=o.color,
                check_sizes=check_sizes,
            )
        else:
            oid = self.load_nonprimitive(
                shape=o.shape,
                position=o.position,
                orientation=o.orientation,
                color=o.color,
                use_fixed_base=use_fixed_base,
            )
        return oid

    def create_primitive(
        self,
        geom: Any,
        position_mode: str,
        position: List[float],
        orientation: List[float],
        r: float,
        h: float,
        color: str = None,
        base_mass: Optional[float] = 3.5,
        check_sizes: Optional[bool] = True,
    ) -> Optional[int]:
        """Creates a primitive object.

        Args:
            geom: The bullet GEOM_{shape} enum.
            position_mode: Whether the position represents the base or the COM.
            position: The (x, y, z) position of the object, in world 
                coordinates.
            orientation: The orientation of the object, in [x, y, z, w] 
                quaternion format.
            r: The radius of the object.
            h: The height of the object. This should be 2*r for sphere.
            color: The color of the object.
            base_mass: The mass of the object.
            check_sizes: Whether to check that the sizes are valid for various
                shapes.

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
        if position_mode == "com":
            com_position = position
        elif position_mode == "base":
            com_position = position.copy()
            com_position[2] += h / 2
        else:
            raise ValueError(f"Invalid position mode: {position_mode}")

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
                f"Shape: {geom}\t"
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
        self,
        shape: str,
        position: List[float],
        orientation: List[float],
        color: str,
        use_fixed_base: Optional[bool] = False,
    ) -> int:
        """Loads a nonprimitive object.

        Args:
            shape: The object shape.
            position: The position to set the origin of the object.
            orientation: The orientation of the object.
            color: The color of the object.
            use_fixed_base: Whether to force the base of the loaded object to
                be static.
        
        Returns:
            oid: The object ID.
        """
        path = self.construct_object_path(obj_name=shape)
        if shape in URDF_SHAPES:
            oid = self.load_urdf(
                path=path,
                position=position,
                orientation=orientation,
                color=color,
                use_fixed_base=use_fixed_base,
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

    def load_urdf(
        self,
        path: str,
        position: List[float],
        orientation: List[float],
        color: str,
        use_fixed_base: Optional[bool] = False,
    ):
        """Loads a urdf object.

        Args:
            path: The path of the urdf file.
            position: The position to set the origin of the object.
            orientation: The orientation of the object.
            color: The color of the object.
            use_fixed_base: Whether to force the base of the loaded object to
                be static.
        
        Returns:
            oid: The object ID.
        """
        oid = self.p.loadURDF(
            path,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=use_fixed_base,
        )
        self.color_object(oid=oid, color=color)
        return oid

    def color_object(self, oid: int, color: str):
        """Applies a color to the object.

        Args:
            oid: The object ID.
            color: The color to apply.
        """
        self.p.changeVisualShape(
            objectUniqueId=oid, linkIndex=-1, rgbaColor=COLOR2RGBA[color]
        )

    def construct_object_path(self, obj_name: str) -> str:
        """Constructs the URDF path based on object attributes.
        
        Args:
            obj_name: The name of the object.
        
        Returns:
            path: The path to the file for the object.
        """
        path = os.path.join(self.assets_dir, SHAPE2PATH[obj_name])
        return path
