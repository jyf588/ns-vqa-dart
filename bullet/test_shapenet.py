import math
import pybullet as p
import time


CAN_HEIGHT_SCALE = 0.24
CAN_RADIUS_SCALE = 0.11
CAN_ORIGIN = [0.0, 0.0, 0.115]

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
shift = [0, 0, 0]
meshScale = [CAN_RADIUS_SCALE * 2, CAN_HEIGHT_SCALE, CAN_RADIUS_SCALE * 2]

visualShapeId = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName="bullet/assets/soda_can.obj",
    rgbaColor=[0.8, 0.0, 0.0, 1.0],
    meshScale=meshScale,
    visualFrameOrientation=[
        math.cos(math.radians(90 / 2)),
        0,
        0,
        math.sin(math.radians(90 / 2)),
    ],
)
collisionShapeId = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName="bullet/assets/soda_can.obj",
    # collisionFramePosition=shift,
    meshScale=meshScale,
)

oid = p.createMultiBody(
    baseCollisionShapeIndex=collisionShapeId,
    baseVisualShapeIndex=visualShapeId,
    basePosition=CAN_ORIGIN,
)

oid = p.loadURDF(
    "bullet/assets/cylinder.urdf",
    # basePosition=[0.0, 0.0, -0.18],
    basePosition=[0.0, 0.0, 0.0],
    useFixedBase=True,
)


for _ in range(10000):
    time.sleep(1 / 240)
    p.stepSimulation()
p.disconnect()
