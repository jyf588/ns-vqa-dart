import numpy as np
import pybullet as p


def create_prim_2_grasp(shape, dim, init_xyz):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder
    # init_xyz vec3 of obj location
    id = None
    if shape == p.GEOM_BOX:
        visualShapeId = p.createVisualShape(shapeType=shape, halfExtents=dim)
        collisionShapeId = p.createCollisionShape(
            shapeType=shape, halfExtents=dim
        )
        id = p.createMultiBody(
            baseMass=3.5,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=init_xyz,
        )
    elif shape == p.GEOM_CYLINDER:
        visualShapeId = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1])
        collisionShapeId = p.createCollisionShape(
            shape, dim[0], [1, 1, 1], dim[1]
        )
        id = p.createMultiBody(
            baseMass=3.5,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=init_xyz,
        )
    elif shape == p.GEOM_SPHERE:
        pass  # TODO
    return id


def main():
    is_box = True
    shape = "box"
    half_height = np.random.uniform(low=0.055, high=0.09)
    half_width = np.random.uniform(low=0.03, high=0.05)  # aka radius
    xyz = [0, 0, 0]

    if shape == "box":
        dim = [half_width * 0.8, half_width * 0.8, half_height]
        obj_id = create_prim_2_grasp(p.GEOM_BOX, dim, xyz)
    elif shape == "cylinder":
        dim = [half_width, half_height * 2.0]  # TODO
        obj_id = create_prim_2_grasp(p.GEOM_CYLINDER, dim, xyz)


if __name__ == "__main__":
    main()
