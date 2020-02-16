import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
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
