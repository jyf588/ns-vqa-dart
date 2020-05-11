import os
import sys
import json
import shutil
import pickle
import pybullet
import numpy as np
from typing import *
from numpy.linalg import inv
from os.path import expanduser
import pybullet_utils.bullet_client as bc
from scipy.spatial.transform import Rotation as R


CONNECT_MODE2FLAG = {"direct": pybullet.DIRECT, "gui": pybullet.GUI}


def create_bullet_client(mode: str) -> bc.BulletClient:
    return bc.BulletClient(connection_mode=CONNECT_MODE2FLAG[mode])


""" Geometry """


def create_transformation(position: List[float], orientation: List[float]):
    """Creates a transformation matrix for the provided position and 
    orientation.

    Args:
        position: The xyz position.
        orientation: The xyzw orientation, in quaternion format.

    Returns:
        transformation: The 4x4 transformation matrix.
    """
    transformation = np.identity(4)
    r = R.from_quat(orientation)
    transformation[:3, :3] = r.as_matrix()
    transformation[:3, 3] = position
    return transformation


def apply_transform(xyz: List[float], transformation: np.ndarray):
    """Applies a transformation to a set of xyz values.

    Args:
        xyz: The xyz values to transform.
        transformation: A 4x4 transformation matrix to apply.

    Returns:
        xyz_transformed: Transformed xyz values.
    """
    vec = np.array(list(xyz) + [1.0])
    vec_transformed = np.dot(transformation, vec)
    xyz_transformed = list(vec_transformed[:3])
    return xyz_transformed


def apply_inv_transform(xyz: List[float], transformation: np.ndarray):
    """Applies an inverse transform to a set of xyz values.

    Args:
        xyz: The xyz values to transform.
        transformation: A 4x4 transformation matrix. The inverse of this
            transformation will be applied.

    Returns:
        xyz_transformed: The xyz values, transformed by the inverse of the 
            input transformation.
    """
    xyz_transformed = apply_transform(xyz=xyz, transformation=inv(transformation))
    return xyz_transformed


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
    world_vec = np.array(list(xyz) + [1.0])
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
    x, y, z = up
    a1_solved = np.arcsin(-y)
    a2_solved = np.arctan2(x, z)
    # a3_solved is zero since equation has under-determined
    if gt_orientation is None:
        orientation = pybullet.getQuaternionFromEuler([a1_solved, a2_solved, 0])
    else:
        raise NotImplementedError

    """
    if gt_orientation is None:
        rotation = np.zeros((3, 3))
        # rotation = np.identity(3)
    else:
        rotation = orientation_to_rotation(orientation=gt_orientation)
        rotation = np.array(rotation).reshape((3, 3))

    # Set the up vector.
    rotation[:, -1] = up

    # Convert to orientation.
    orientation = rotation_to_orientation(rotation=rotation)
    """
    return orientation


def up_to_euler(up: List[float]):
    """Converts from an up vector into euler angles.

    Args:
        up: An up vector.
    
    Returns:
        euler: Euler xyz angles (degrees) representing the provided up vector.
    """
    # rotmat = np.identity(3)
    rotmat = np.zeros((3, 3))
    rotmat[:, -1] = up
    r = R.from_matrix(rotmat)
    euler = r.as_euler("xyz", degrees=True)
    # orn = up_to_orientation(up=up)
    # r = R.from_quat(orn)
    # euler = r.as_euler("xyz", degrees=True)
    return euler


def orientation_to_rotation(orientation: List[float]) -> List[float]:
    """Converts an orientation vector into a rotation matrix.

    Args:
        orientation: The quaternion representing the orientation. [x, y, z, w].
    
    Returns:
        rotation: The 3x3 rotation matrix.
    """
    # p = create_bullet_client(mode="direct")
    # rotation = pybullet.getMatrixFromQuaternion(quaternion=orientation)
    r = R.from_quat(orientation)
    rotation = r.as_matrix()
    return rotation


def orientation_to_euler(orientation: List[float]) -> List[float]:
    """Converts an orientation into euler angles.

    Args:
        orientation: The quaternion representing the orientation. [x, y, z, w].
    
    Returns:
        euler_angles: The euler angle representation of the input quaternion.
    """
    q = R.from_quat(orientation)
    euler_angles = q.as_euler("xyz", degrees=True)
    return list(euler_angles)


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


def rotation_to_orientation(rotation: List[float]) -> List[float]:
    """Converts a rotation matrix into a quaternion.

    Args:
        rotation: The 3x3 rotation matrix.

    Returns:
        orientation: The [x, y, z, w] quaternion.
    """
    orientation = R.from_matrix(np.array(rotation).reshape((3, 3))).as_quat()
    return list(orientation)


def euler_to_up(euler: List[float]) -> List[float]:
    """Converts euler angles into an up vector.

    Args:
        euler: A set of xyz euler angles (degrees).
    
    Returns:
        up: The up vector corresponding to the euler angles.
    """
    r = R.from_euler("xyz", euler, degrees=True)
    rotmat = r.as_matrix()
    up = rotation_to_up(rotation=rotmat)
    return up


""" File I/O utility functions. """


def delete_and_create_dir(dir: str):
    if os.path.exists(dir):
        user_input = input(f"dst dir already exists: {dir}. Delete and continue? [Y/n]")
        if user_input == "Y":
            shutil.rmtree(dir)
        else:
            print(f"user_input: {user_input}. Exiting.")
            sys.exit(0)
    os.makedirs(dir)


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
    assert not os.path.exists(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def get_user_homedir() -> str:
    home = expanduser("~")
    return home
