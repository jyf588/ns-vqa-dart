import pybullet as p
import pybullet_data
import time


def main():
    # # Initialize the Bullet client and renderer.
    # self.p = bullet.util.create_bullet_client(mode=render_mode)
    # self.renderer = BulletRenderer(p=self.p)

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    boxId = p.loadURDF("bullet/assets/lego.urdf")
    for _ in range(10000):
        time.sleep(1 / 240)
    p.disconnect()


if __name__ == "__main__":
    main()
