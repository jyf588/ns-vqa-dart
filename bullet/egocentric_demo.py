import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
import time

from vision.camera import BulletCamera
from vision.renderer import BulletRenderer

ROBOT_URDF_PATH = "my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_revolute_head.urdf"

"""
<joint name="head_pan_joint" type="revolute">
    <axis xyz="0 0 1"/>
    <limit effort="1000.0" lower="-1.57079632679" upper="1.57079632679" velocity="1745.32925199"/>
    <origin rpy="0 0 0" xyz="0 0 0.0245"/>
    <parent link="head_base_link"/>
    <child link="head_link"/>
</joint>

<joint name="r_elbow_roll_joint" type="revolute">
    <axis xyz="0 0 1"/>
    <limit effort="1000.0" lower="-1.57079632679" upper="1.57079632679" velocity="1745.32925199"/>
    <origin rpy="0 0 0" xyz="0.0 0 0.0"/>
    <parent link="r_forearm_link_aux"/>
    <child link="r_forearm_link"/>
</joint>
"""


# HEAD_ROLL_JOINT = 8  # z rot (left/right)
# HEAD_TILT_JOINT = 9  # x rot (left/right ear)
# HEAD_PAN_JOINT = 10  # y rot (up/down)


HEAD_ROLL_JOINT = 8  # x rot (left/right ear)
HEAD_TILT_JOINT = 9  # y (up/down)
# HEAD_PAN_JOINT = 10  # z rot (look left/right)

AXIS2IDX = {"x": 0, "y": 1, "z": 2}


def main():
    camera = BulletCamera(p=p)
    renderer = BulletRenderer(p=p)
    p.connect(p.GUI)
    robot_id = p.loadURDF(
        ROBOT_URDF_PATH,
        -0.300000,
        0.2,  # 0.500000,
        -1.250000,
        0.000000,
        0.000000,
        0.000000,
        1.000000,
    )
    renderer.render_table()
    renderer.render_object(
        position=[0.2, 0.2, 0.1],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        size="large",
        shape="cylinder",
        color="yellow",
        fix_base=True,
    )
    renderer.render_object(
        position=[0.15, 0.35, 0.1],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        size="large",
        shape="box",
        color="red",
        fix_base=True,
    )
    renderer.render_object(
        position=[0.10, 0.05, 0.1],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        size="large",
        shape="cylinder",
        color="green",
        fix_base=True,
    )
    renderer.render_object(
        position=[0.2, -0.05, 0.1],
        quaternion=[0.0, 0.0, 0.0, 1.0],
        size="large",
        shape="box",
        color="blue",
        fix_base=True,
    )

    for joint_idx in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_name = joint_info[1].decode("utf-8")
        print(joint_name)
        if joint_name == "head_roll_joint":
            head_roll_id = joint_info[0]
        elif joint_name == "head_tilt_joint":
            head_tilt_id = joint_info[0]
        elif joint_name == "eyes_pan_joint":
            eyes_pan_id = joint_info[0]
        elif joint_name == "eyes_tilt_joint":
            eyes_tilt_id = joint_info[0]
        elif joint_name == "r_eye_joint":
            r_eye_id = joint_info[0]
        elif joint_name == "r_iris_joint":
            r_iris_id = joint_info[0]

    tilt_degrees = list(range(45, 50, 1)) + list(range(50, 45, -1))
    # tilt_degrees = []
    roll_degrees = (
        list(range(0, 10, 1))
        + list(range(10, -10, -1))
        + list(range(-10, 0, 1))
    )
    degrees_seq = []

    for tilt_d in tilt_degrees:
        degrees_seq.append({"tilt": tilt_d, "roll": 0.0})
    for roll_d in roll_degrees:
        degrees_seq.append({"tilt": 45.0, "roll": roll_d})

    name2info = {
        "tilt": {"joint_id": head_tilt_id, "axis": "y"},
        "roll": {"joint_id": head_roll_id, "axis": "x"},
    }

    for i in range(10000):
        rotation_angles = np.array([0.0, 0.0, 0.0])
        for name, d in degrees_seq[i % len(degrees_seq)].items():
            joint_id = name2info[name]["joint_id"]
            axis = name2info[name]["axis"]

            # Set the orientation of the joint.
            radians = math.radians(d)
            p.resetJointState(robot_id, joint_id, radians)
            rotation_angles[AXIS2IDX[axis]] = d

        rot = R.from_euler("xyz", rotation_angles, degrees=True)

        position = np.array(p.getLinkState(robot_id, eyes_pan_id)[0])
        position2cam = np.array([0.05, 0.0, 0.0])
        position2cam = rot.apply(position2cam)
        cam_position = position + position2cam

        # Compute the target position.
        cam2target = np.array([1.0, 0.0, 0.0])
        cam2target = rot.apply(cam2target)
        target_position = cam_position + cam2target

        up_vec = np.array([0.0, 0.0, 1.0])
        up_vec = rot.apply(up_vec)
        up_vec[1] = -1 * up_vec[1]

        camera.set_cam_position(cam_position, target_position, up_vec=up_vec)
        rgb, mask = camera.get_rgb_and_mask()

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
