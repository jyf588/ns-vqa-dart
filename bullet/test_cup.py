import numpy as np
import pybullet as p
import pybullet_data
import time

from bullet.aabb import drawAABB


def main():
    """
    [-0.0903642 -0.0561729 -0.0048801]
    [0.0374984  0.0745419  0.18576861]
    origin: (0.064)
    scale: (.5 .5 .5)
    new scale: (1. 1. 1.3)

    width: 0.0
    """
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    oid = p.loadURDF("bullet/assets/cup/cup_small.urdf", basePosition=[0., 0., 0.], useFixedBase=True)
    # oid = p.loadURDF("bullet/assets/box.urdf", basePosition=[0., 0., 0.], useFixedBase=True)

    aabb = p.getAABB(oid)
    aabbMin = np.array(aabb[0])
    aabbMax = np.array(aabb[1])

    # aabbMin += 0.003
    # aabbMax -= 0.003
    # aabb = (list(aabbMin), list(aabbMax))

    print(aabbMin)
    print(aabbMax)
    drawAABB(aabb)

    for _ in range(10000):
        time.sleep(1 / 240)
        p.stepSimulation()
    p.disconnect()



if __name__ == "__main__":
    main()
