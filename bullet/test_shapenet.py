import math
import pybullet as p
import time


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
shift = [0, 0, 0]
meshScale = [1.0, 1.0, 1.0]

visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="bullet/assets/model_normalized.obj",
    rgbaColor=[0.8, 0.0, 0.0, 1.0],
    visualFrameOrientation=[
        math.cos(math.radians(90 / 2)),
        0,
        0,
        math.sin(math.radians(90 / 2)),
    ],
)
collisionShapeId = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="bullet/assets/model_normalized.obj",
    # collisionFramePosition=shift,
    # meshScale=meshScale,
)

oid = p.createMultiBody(
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=[0.0, 0.0, 0.5],
)


for _ in range(10000):
    time.sleep(1 / 240)
    p.stepSimulation()
p.disconnect()
