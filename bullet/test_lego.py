import pybullet as p
import pybullet_data
import time


def main():
    # # Initialize the Bullet client and renderer.
    # self.p = bullet.util.create_bullet_client(mode=render_mode)
    # self.renderer = BulletRenderer(p=self.p)

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    oid = p.loadURDF("bullet/assets/lego_small.urdf", basePosition=[0, 0, 0])
    oid = p.loadURDF(
        "bullet/assets/box_small.urdf", basePosition=[0.025, 0.15, 0]
    )
    oid = p.loadURDF("bullet/assets/lego.urdf", basePosition=[0, 0.25, 0])
    oid = p.loadURDF("bullet/assets/box.urdf", basePosition=[0.04, 0.45, 0])
    # lego = p.loadURDF(
    #     "bullet/assets/lego.urdf", basePosition=[0, 0, 0], useFixedBase=True
    # )
    # box = p.loadURDF(
    #     "bullet/assets/box.urdf",
    #     basePosition=[0.04, 0.04, -0.1],
    #     useFixedBase=True,
    # )
    """
    x2.35
    7.2
    Lego: 
        (-0.017, 0.017)
        (-0.017, 0.017)
        (-0.0125, 0.0125)
    Box:
        (-0.04, 0.04)
        (-0.04, 0.04)
        (-0.09, 0.09)
    """
    # min_aabb, max_aabb = p.getAABB(lego)
    # print(min_aabb)
    # print(max_aabb)
    for _ in range(10000):
        time.sleep(1 / 240)
        p.stepSimulation()
    p.disconnect()


if __name__ == "__main__":
    main()
