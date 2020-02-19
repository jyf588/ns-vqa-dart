"""
https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createMultiBodyLinks.py
"""

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
    "box": {"width": 0.8, "height": 1.0},
    "cylinder": {"width": 1.0, "height": 2.0},
    "sphere": {"width": 1.0, "height": 1.0},
}


def create_prim_2_grasp(geom, dim: List[float], position: List[float]):
    """Creates a primitive object.

    Args:
        geom: The pybullet shape type.
        dim: A list of dimensions, of variable length / contents depending on
            the shape.
        position: The (x, y, z) position of the COM.
    """
    if geom == p.GEOM_BOX:
        visualShapeId = p.createVisualShape(shapeType=geom, halfExtents=dim)
        collisionShapeId = p.createCollisionShape(
            shapeType=geom, halfExtents=dim
        )
    elif geom == p.GEOM_CYLINDER:
        visualShapeId = p.createVisualShape(geom, dim[0], [1, 1, 1], dim[1])
        collisionShapeId = p.createCollisionShape(
            geom, dim[0], [1, 1, 1], dim[1]
        )

    elif geom == p.GEOM_SPHERE:
        visualShapeId = p.createCollisionShape(shapeType=geom, radius=dim[0])
        collisionShapeId = p.createCollisionShape(geom, radius=dim[0])

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
    # shapes = ["sphere"]

    half_height = np.random.uniform(low=0.055, high=0.09)
    half_width = np.random.uniform(low=0.03, high=0.05)  # aka radius

    for i, shape in enumerate(shapes):
        half_width *= SHAPE2MULTIPLIER[shape]["width"]
        half_height *= SHAPE2MULTIPLIER[shape]["height"]
        if shape == "box":
            dim = [half_width, half_width, half_height]
        elif shape == "cylinder":
            dim = [half_width, half_height]
        elif shape == "sphere":
            dim = [half_width]
        print(f"width: {half_width}")
        print(f"height: {half_height}")
        obj_id = create_prim_2_grasp(SHAPE2GEOM[shape], dim, [0, 0.1 * i, 0])

    for _ in range(10000):
        p.stepSimulation()
        time.sleep(1 / 240)


if __name__ == "__main__":
    main()
