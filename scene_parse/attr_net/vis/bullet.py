import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R


COLORS = {"red": [0.8, 0.0, 0.0, 1.0],
          "grey": [0.4, 0.4, 0.4, 1.0],
          "yellow": [0.8, 0.8, 0.0, 1.0],
          "blue": [0.0, 0.0, 0.8, 1.0],
          "green": [0.0, 0.8, 0.0, 1.0]}

name2urdf = {
    'large_box': 'assets/box.urdf',
    'large_cylinder': 'assets/cylinder.urdf',
    'tabletop': 'assets/tabletop.urdf'
}

def render_scene(scene_objects, table):
    if table:
        scene_objects.append({
            'shape': 'tabletop',
            'size': None,
            'color': 'grey',
            'position': [0.25, 0.2, 0.0],
            'rotation': None
        })
    for obj in scene_objects:
        size = obj['size']
        shape = obj['shape']
        name = shape if size is None else f"{size}_{shape}"
        render_bullet_object(
            name=name,
            position=obj['position'],
            rotation=obj['rotation'],
            color=obj['color']
        )
    img = render_image()
    p.resetSimulation()
    return img


def render_bullet_object(name, position, rotation=None, color=None):
    # Convert 3x3 rotation matrix to quaternion.
    urdf_path = name2urdf[name]
    if rotation is None:
        obj_id = p.loadURDF(urdf_path, position, useFixedBase=-1)
    else:
        quaternion = R.from_matrix(np.array(rotation).reshape((3, 3))).as_quat()
        obj_id = p.loadURDF(urdf_path, position, quaternion, useFixedBase=-1)
    if color is not None:
        p.changeVisualShape(obj_id, -1, rgbaColor=COLORS[color])

def render_image():
    H, W, view_mat, proj_mat = init_camera()
    img = p.getCameraImage(H, W, viewMatrix=view_mat, projectionMatrix=proj_mat, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.reshape(img[2], (W, H, 4))[:, :, :3]  # alpha channel dropped
    # mask = np.reshape(img[4], (self.W, self.H))
    return rgb

def init_camera():
    H = 480
    W = 320
    cam_target_pos = [0.25, 0.2, 0]  # [0, 0, 0]
    cam_distance = 0.81
    pitch = -30.0
    roll = 0
    yaw = 270
    up_axis_index = 2
    proj_mat = [
        1.0825318098068237,
        0.0,
        0.0, 
        0.0,
        0.0,
        1.732050895690918,
        0.0,
        0.0,
        0.0,
        0.0,
        -1.0002000331878662,
        -1.0,
        0.0,
        0.0,
        -0.020002000033855438,
        0.0
    ]
    view_mat = p.computeViewMatrixFromYawPitchRoll(
        cam_target_pos, 
        cam_distance, 
        yaw, 
        pitch, 
        roll,
        up_axis_index
    )
    return H, W, view_mat, proj_mat
