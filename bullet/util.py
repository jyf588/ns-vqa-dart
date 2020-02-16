import json
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation as R
from typing import *


CONNECT_MODE2FLAG = {"direct": pybullet.DIRECT, "gui": pybullet.GUI}


def create_bullet_client(mode: str) -> bc.BulletClient:
    return bc.BulletClient(connection_mode=CONNECT_MODE2FLAG[mode])


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


def orientation_to_rotation(orientation: List[float]) -> List[float]:
    """Converts an orientation vector into a rotation matrix.

    Args:
        orientation: The quaternion representing the orientation. [x, y, z, w].
    
    Returns:
        rotation: The 3x3 rotation matrix.
    """
    p = create_bullet_client(mode="direct")
    rotation = p.getMatrixFromQuaternion(quaternion=orientation)
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


""" JSON functions. """


def load_json(path: str) -> Any:
    """Loads a JSON file.

    Args:
        path: The path to the JSON file.
    
    Returns:
        data: The JSON data.
    """
    with open(path, "r") as f:
        data = json.load(f)
        return data


def save_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, sort_keys=True, indent=2, separators=(",", ": "))
