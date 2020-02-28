import pybullet as p
import time


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
shift = [0, 0, 0]
meshScale = [1.0, 1.0, 1.0]

visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="bullet/assets/model_normalized.obj",
    rgbaColor=[1, 1, 1, 1],
    specularColor=[0.4, 0.4, 0],
    visualFramePosition=shift,
    meshScale=meshScale,
)
collisionShapeId = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="bullet/assets/model_normalized.obj",
    collisionFramePosition=shift,
    meshScale=meshScale,
)

p.createMultiBody(
    baseMass=1,
    baseInertialFramePosition=[0, 0, 0],
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0.0, 0.0, 0.0],
    useMaximalCoordinates=True,
)


for _ in range(10000):
    time.sleep(1 / 240)
    p.stepSimulation()
p.disconnect()
