import json
import numpy as np
from numpy.linalg import inv
import os
import pickle
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation as R
from typing import *


CONNECT_MODE2FLAG = {"direct": pybullet.DIRECT, "gui": pybullet.GUI}


def create_bullet_client(mode: str) -> bc.BulletClient:
    return bc.BulletClient(connection_mode=CONNECT_MODE2FLAG[mode])


""" Geometry """


def world_to_cam(xyz: List[float], camera) -> List[float]:
    """Converts xyz coordinates from world to camera coordinate frame.

    Args:
        xyz: The xyz point.
        camera: A BulletCamera.
    
    Returns:
        cam_xyz: The xyz point in camera coordinate frame.
    """
    view_mat = np.array(camera.view_mat)
    view_mat = view_mat.reshape((4, 4))
    world_vec = np.array(xyz + [1.0])
    cam_vec = np.dot(view_mat, world_vec)
    cam_xyz = list(cam_vec[:3])
    return cam_xyz


def cam_to_world(xyz: List[float], camera) -> List[float]:
    """Converts xyz coordinates from camera to world coordinate frame.

    Args:
        xyz: The xyz point.
        camera: A BulletCamera.
    
    Returns:
        world_xyz: The xyz point in world coordinate frame.
    """
    assert type(xyz) == list

    view_mat = np.array(camera.view_mat)
    view_mat = view_mat.reshape((4, 4))
    cam_vec = np.array(xyz + [1.0])
    world_vec = np.dot(inv(view_mat), cam_vec)
    world_xyz = list(world_vec[:3])
    return world_xyz


def orientation_to_up(orientation: List[float]) -> List[float]:
    """Extracts the up vector from the orientation.

    Args:
        orientation: The quaternion representing the orientation. [x, y, z, w].
    
    Returns:
        up: The up vector.
    """
    rotation = orientation_to_rotation(orientation=orientation)
    up = rotation_to_up(rotation)
    return up


def up_to_orientation(
    up: List[float], gt_orientation: Optional[List[float]] = None
) -> List[float]:
    """Converts an up vector to an orientation, in quaternion format.

    Args:
        up: The up vector.
        gt_orientation: If supplied, uses the x and y rotation from GT to
            compose the final orientation.
    
    Returns:
        orientation: The orientation in xyzw quaternion format.
    """
    if gt_orientation is None:
        rotation = np.zeros((3, 3))
    else:
        rotation = orientation_to_rotation(orientation=gt_orientation)
        rotation = np.array(rotation).reshape((3, 3))

    # Set the up vector.
    rotation[:, -1] = up

    # Convert to orientation.
    orientation = rotation_to_quaternion(rotation=rotation)
    return orientation


def orientation_to_rotation(orientation: List[float]) -> List[float]:
    """Converts an orientation vector into a rotation matrix.

    Args:
        orientation: The quaternion representing the orientation. [x, y, z, w].
    
    Returns:
        rotation: The 3x3 rotation matrix.
    """
    # p = create_bullet_client(mode="direct")
    rotation = pybullet.getMatrixFromQuaternion(quaternion=orientation)
    return rotation


def rotation_to_up(rotation: List[float]) -> List[float]:
    """Extracts the up vector from a rotation matrix.

    Args:
        rotation: The 3x3 rotation matrix.
    
    Returns:
        up: The up vector.
    """
    rotation = np.array(rotation).reshape((3, 3))
    up = list(rotation[:, -1])
    return up


def rotation_to_quaternion(rotation: List[float]) -> List[float]:
    """Converts a rotation matrix into a quaternion.

    Args:
        rotation: The 3x3 rotation matrix.

    Returns:
        quaternion: The [x, y, z, w] quaternion.
    """
    quaternion = R.from_matrix(np.array(rotation).reshape((3, 3))).as_quat()
    return list(quaternion)


""" JSON and pickle I/O utility functions. """


def load_json(path: str) -> Any:
    """Loads a JSON file.

    Args:
        path: The path to the JSON file.
    
    Returns:
        data: The JSON data.
    """
    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"JSONDecodeError for file: {path}")
            raise (e)
        return data


def save_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=2, separators=(",", ": "))


def load_pickle(path: str) -> Any:
    """
    Args:
        path: The path of the pickle file to load.
    
    Returns:
        data: The data in the pickle file.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(path: str, data: Any):
    """
    Args:
        path: The path of the pickle file to save.
        data: The data to save.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
