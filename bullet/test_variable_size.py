"""
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createVisualShape.py
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createMultiBodyLinks.py
"""
import inspect
import numpy as np
import pybullet as p
import time
from typing import *


SHAPE2GEOM = {
    "box": p.GEOM_BOX,
    "cylinder": p.GEOM_CYLINDER,
    "sphere": p.GEOM_SPHERE,
}

SHAPE2MULTIPLIER = {
    "box": {"r": 0.8},
    "cylinder": {"r": 1.0},
    "sphere": {"r": 1.0},
}


def generate_primitive_shape(
    shape: str, position: List[float], r: float, h: Optional[float] = None
):
    """Creates a primitive object.

    Args:
        shape: The name of the shape to generate.
        r: The radius of the object.
        h: The height of the object. Unused for sphere.
        dim: A list of dimensions, of variable length / contents depending on
            the shape:
                box: [radius, radius, height]
                cylinder: [radius, half_height]
                sphere: [radius]
        position: The (x, y, z) position of the COM, in world coordinates.
    """
    geom = SHAPE2GEOM[shape]
    half_extents = [r, r, h / 2]
    # print(inspect.signature(p.createVisualShape))
    if geom == p.GEOM_BOX:
        visualShapeId = p.createVisualShape(
            shapeType=geom, halfExtents=half_extents
        )
        collisionShapeId = p.createCollisionShape(
            shapeType=geom, halfExtents=half_extents
        )
    elif geom == p.GEOM_CYLINDER:
        half_extents = [1, 1, 1]
    elif geom == p.GEOM_SPHERE:
        half_extents = [1, 1, 1]
    else:
        raise ValueError(f"Invalid geometry: {geom} for shape: {shape}.")

    visualShapeId = p.createVisualShape(
        shapeType=geom, radius=r, halfExtents=half_extents, length=h
    )
    collisionShapeId = p.createCollisionShape(
        shapeType=geom, radius=r, halfExtents=half_extents, height=h
    )
    p.createMultiBody(
        baseMass=3.5,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=position,
    )


def main():
    p.connect(p.GUI)

    shapes = ["box", "cylinder", "sphere"]
    """
    box:
        Small: (xy=0.05, h=0.13)
        Large: (xy=0.08, h=0.18)
    cylinder:
        Small: (r=0.04, h=0.13)
        Large: (r=0.05, h=0.18)
    """
    orig_shapes = [
        {"shape": "cylinder", "r": 0.04, "h": 0.13, "urdf": "cylinder"},
        {"shape": "box", "r": 0.04, "h": 0.18, "urdf": "box"},
        {"shape": "cylinder", "r": 0.05, "h": 0.18, "urdf": "cylinder_small"},
        {"shape": "box", "r": 0.025, "h": 0.13, "urdf": "box_small"},
    ]

    h_min, h_max = 0.11, 0.18
    r_min, r_max = 0.03, 0.05

    h = np.random.uniform(low=0.11, high=0.18)
    r = np.random.uniform(low=0.03, high=0.05)

    h_interval = (h_max - h_min) / 5
    r_interval = (r_max - r_min) / 5

    y = 0
    for shape in shapes:
        print(f"Shape: {shape}")
        for i in range(6):
            h = h_min + i * h_interval
            r = r_min + i * r_interval

            print(f"h: {h}")
            print(f"r: {r}")

            r *= SHAPE2MULTIPLIER[shape]["r"]

            z = r if shape == "sphere" else h / 2
            generate_primitive_shape(shape=shape, r=r, h=h, position=[0, y, z])
            y += 0.1

    print(f"***Original shapes***")

    y = -0.2
    for shape_dict in orig_shapes:
        # shape = shape_dict["shape"]
        # r = shape_dict["r"]
        # h = shape_dict["h"]
        # z = h / 2
        # generate_primitive_shape(shape=shape, r=r, h=h, position=[0, y, z])
        # y -= 0.15
        # print(f"Shape: {shape}")
        # print(f"h: {h}")
        # print(f"r: {r}")

        urdf = shape_dict["urdf"]
        oid = p.loadURDF(
            f"bullet/assets/{urdf}.urdf",
            basePosition=[0.0, y, 0],
            useFixedBase=True,
        )
        p.changeVisualShape(oid, -1, rgbaColor=[0.8, 0.8, 0.0, 1.0])
        y -= 0.15

    # p.loadURDF(
    #     "bullet/assets/box.urdf", basePosition=[0.3, 0, 0], useFixedBase=True
    # )

    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()
