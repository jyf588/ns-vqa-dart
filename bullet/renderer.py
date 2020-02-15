"""
Definition of a BulletRenderer which renders various objects in a bullet scene.
"""

import math
import numpy as np
import os
import pybullet
import pybullet_utils.bullet_client as bc

from bullet.camera import BulletCamera
from bullet.dash_object import DashObject

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
        self.camera = BulletCamera(p=self.p)

        # Note: z position is defined by the top of the table.
        self.table_attr = {
            "shape": "tabletop",
            "color": "grey",
            "size": None,
            "position": [
                0.25,
                0.2,
                0.0,  # 0.0,
            ],  # ],  # This is COM, base is [0.25, 0.2, 0.0]
            "quaternion": [0, 0, 0, 1],
        }

    def render_scene(self, pose_dict, oid2attr):
        self.p.resetSimulation()
        for oid, pose in pose_dict.items():
            pose.update(oid2attr[oid])
            self.render_object(
                position=pose["position"],
                quaternion=pose["quaternion"],
                size=oid2attr[oid]["size"],
                shape=oid2attr[oid]["shape"],
                color=oid2attr[oid]["color"],
            )
        self.render_table()
        rgb, mask = self.camera.get_rgb_and_mask()
        return rgb

    def render_table(self):
        self.render_object(
            position=self.table_attr["position"],
            quaternion=self.table_attr["quaternion"],
            size=self.table_attr["size"],
            shape=self.table_attr["shape"],
            color=self.table_attr["color"],
            fix_base=True,
        )

    def render_object(self, o: DashObject, fix_base=False):
        # We set the pose in resetBasePositionAndOrientation instead of
        # loadURDF because loadURDF sets the pose at the base, while
        # resetBasePositionAndOrientation sets the pose at the COM. We want COM
        # because the attributes were stored using the
        # getBasePositionAndOrientation function, which returns COM
        # coordinates.
        oid = self.p.loadURDF(
            fileName=self.construct_urdf_path(o=o),
            basePosition=o.world_position,
            baseOrientation=o.world_orientation,
            useFixedBase=fix_base,
        )
        self.p.changeVisualShape(
            objectUniqueId=oid, linkIndex=-1, rgbaColor=COLOR2RGBA[o.color]
        )

    def construct_urdf_path(self, o: DashObject) -> str:
        """Constructs the URDF path based on object attributes.
        
        Args:
            o: A DashObject.
        
        Returns:
            urdf_path: The path to the urdf file matching the attributes of the
                DashObject.
        """
        if o.size in [None, "large"]:
            urdf_name = o.shape
        elif o.size == "small":
            urdf_name = f"{o.shape}_{o.size}"
        else:
            raise ValueError(f"Unsupported size: {o.size}.")

        urdf_path = os.path.join(self.urdf_dir, f"{urdf_name}.urdf")
        return urdf_path
