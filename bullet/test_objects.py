import pybullet as p
import pybullet_data
import time


def main():
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath("/Users/michelleguo/workspace/third_party/bulletphysics/bullet3/data")  # optionally
    # oid = p.loadURDF("dinnerware/cup/cup_small.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("/Users/michelleguo/workspace/third_party/bulletphysics/bullet3/data/dinnerware/pan_tefal.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("dinnerware/plate.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("dinnerware/pan_tefal.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("dinnerware/cup/cup_small.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("wheel.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("capsule.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("cube_soft.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("husky/duck_vhacd.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("husky/husky.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("jenga/jenga.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("marble_cube.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("teddy_large.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("teddy_vhacd.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("torus/torus.urdf", basePosition=[0, 0, 0])
    # oid = p.loadURDF("racecar/racecar.urdf", basePosition=[0, 0, 0])
    oid = p.loadSDF("kitchens/1.sdf")
    for _ in range(10000):
        time.sleep(1 / 240)
        p.stepSimulation()
    p.disconnect()


if __name__ == "__main__":
    main()
