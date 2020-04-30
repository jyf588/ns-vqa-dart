"""Generates a dataset that can be used for downstream training and evaluation.

Expects the following file structure for <args.states_dir>:
    <args.states_dir>/
        <sid>.p = {
            "objects": {
                <oid>: {
                    "shape": <shape>,
                    "color": <color>,
                    "radius": <radius>,
                    "height": <height>,
                    "position": <position>,
                    "orientation": <orientation
                }
            }
            "robot": {<joint_name>: <joint_angle>}
        }

Expects the following file structure for <args.image_dir>:
    <args.img_dir>/
        img/
            <sid>_<oid>.png
            ...
        id/
            <sid>_<oid>.png
            ...
# Note that IDs in the mask must correspond to the IDs in object states.

Generates the dataset to <args.dst_dir> in the following structure/format:
    <args.dst_dir>/
        partition.json
        data/
            <sid>_<oid>.p = [
                X,      # Input data (np.ndarray), shape (H, W, 6)
                y,      # Labels (np.ndarray), shape (n_labels,)
                sid,    # Scene ID of the example.
                oid,    # Object ID of the example.
                path    # The path of the current pickle file.
            ]
"""
import argparse
import cv2
import imageio
import numpy as np
import os
import pprint
from tqdm import tqdm
from typing import *

from ns_vqa_dart.bullet.dash_dataset import DashDataset
import ns_vqa_dart.bullet.dash_object as dash_object
import ns_vqa_dart.bullet.random_objects as random_objects
import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    util.delete_and_create_dir(dir=args.dst_dir)
    # os.makedirs(args.dst_dir, exist_ok=True)

    paths = []
    # Loop over the scene IDs.
    for sid in tqdm(range(args.start_sid, args.end_sid)):
        # Load the state for the current scene ID.
        state = util.load_pickle(
            path=os.path.join(args.states_dir, f"{sid:06}.p")
        )

        oids = list(state["objects"].keys())

        # Only generate for the first two objects if the camera control is
        # stacking. Here we are assuming that the first two objects in the
        # state dictionary always corresponds to the two objects in the stack.
        if args.objects_to_include == 2:
            oids = oids[:2]
        elif args.objects_to_include == 1:
            oids = oids[:1]
        elif args.objects_to_include == -1:
            pass
        else:
            raise ValueError(
                f"Invalid number of objects to include: {args.objects_to_include}"
            )

        # Save the input data.
        for oid in oids:
            odict = state["objects"][oid]
            X, y = load_X_and_y(
                sid=sid,
                oid=oid,
                odict=odict,
                img_dir=args.img_dir,
                cam_dir=args.cam_dir,
                camera_control=args.camera_control,
                coordinate_frame=args.coordinate_frame,
            )

            # Skip over the example if there are issues with the example (e.g.,
            # object is completely occluded)
            if X is None and y is None:
                print(f"Skipping occluded example sid: {sid}\toid: {oid}")
                continue

            path = save_example(
                data_dir=args.dst_dir, sid=sid, oid=oid, X=X, y=y
            )
            paths.append(path)

    # Save a partition of the data.
    split_id = int(len(paths) * 0.8)
    partition = {
        "train": paths[:split_id],
        "val": paths[split_id:],
        "test": paths[split_id:],
    }
    util.save_json(
        path=os.path.join(args.dst_dir, "partition.json"), data=partition
    )
    train = len(partition["train"])
    val = len(partition["val"])
    test = len(partition["test"])
    print(f"Saved partition. Train: {train}\tValidation: {val}\tTest: {test}")


def load_X_and_y(
    sid: int,
    oid: int,
    odict: Dict,
    img_dir: str,
    cam_dir: str,
    camera_control: str,
    coordinate_frame: str,
):

    # Load the image and segmentation for the object.
    rgb, seg = load_rgb_and_seg(
        img_dir=img_dir, sid=sid, oid=oid, camera_control=camera_control
    )

    # TODO: Exclude example if mask area is zero.
    X = dash_object.compute_X(oid=oid, img=rgb, seg=seg, keep_occluded=False)

    # The object is completely occluded so we throw the example out.
    if X is None:
        return None, None

    # Prepare camera parameters if we are using camera coordinate
    # frame.
    if coordinate_frame in ["camera", "unity_camera"]:
        cam_position, cam_orientation = load_camera_pose(
            cam_dir=cam_dir, sid=sid, oid=oid, camera_control=camera_control
        )
    else:
        raise ValueError(f"Invalid coordinate frame: {coordinate_frame}")
    y = dash_object.compute_y(
        odict=odict,
        coordinate_frame=coordinate_frame,
        cam_position=cam_position,
        cam_orientation=cam_orientation,
    )
    return X, y


def load_rgb_and_seg(
    img_dir: str, sid: int, oid: int, camera_control: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the RGB image and segmentation map for a given scene.

    Args:
        img_dir: The directory containing the images with the structure:
            <img_dir>/
                img/
                    <sid>_<oid>.png
                    ...
                id/
                    <sid>_<oid>.png
                    ...
        sid: The scene ID.
        oid: The object ID.

    Returns:
        rgb: The RGB image, of shape (H, W, 3)
        segmentation: A 2D segmentation map where each pixel stores the object 
            ID it belongs to.
    """
    rgb, seg_img = load_rgb_and_seg_img(
        img_dir=img_dir, sid=sid, oid=oid, camera_control=camera_control
    )

    # Convert the segmentation image into an array.
    # TODO: Do this before saving the segmentation.
    seg = seg_img_to_map(seg_img)
    return rgb, seg


def load_rgb_and_seg_img(
    img_dir: str, sid: int, oid: int, camera_control: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the RGB image and segmentation map for a given scene.

    Args:
        img_dir: The directory containing the images with the structure:
            <img_dir>/
                img/
                    <sid>_<oid>.png
                    ...
                id/
                    <sid>_<oid>.png
                    ...
        sid: The scene ID.
        oid: The object ID.

    Returns:
        rgb: The RGB image, of shape (H, W, 3)
        segmentation: A 2D segmentation map where each pixel stores the object 
            ID it belongs to.
    """
    cam_tid = get_camera_target_id(oid=oid, camera_control=camera_control)
    rgb_path = os.path.join(img_dir, "first/rgb", f"{sid:06}_{cam_tid}.png")
    seg_path = os.path.join(img_dir, "first/seg", f"{sid:06}_{cam_tid}.png")
    rgb = imageio.imread(uri=rgb_path)
    seg_img = imageio.imread(uri=seg_path)
    return rgb, seg_img


def load_third_person_image(img_dir: str, sid: str):
    path = os.path.join(img_dir, "third/rgb", f"{sid:06}_.png")
    img = imageio.imread(path)
    return img


def get_camera_target_id(oid: int, camera_control: str):
    if camera_control == "all":
        cam_tid = oid
    elif camera_control in ["center", "stack", "position"]:
        cam_tid = 0
    else:
        raise ValueError(f"Invalid camera control {camera_control}")
    return cam_tid


def load_camera_pose(cam_dir: str, sid: int, oid: int, camera_control: str):
    """Creates a camera with same parameters as camera used to capture images
    for the specified object in the specified scene.

    Args:
        cam_dir: The directory containing camera JSON files.
        sid: The sensor ID.
        oid: The object ID.
    
    Returns:
        cam: A BulletCamera.
    """
    cam_tid = get_camera_target_id(oid=oid, camera_control=camera_control)
    path = os.path.join(cam_dir, f"{sid:06}.json")
    params = util.load_json(path=path)[f"{cam_tid}"]
    position = params["camera_position"]
    orientation = params["camera_orientation"]
    return position, orientation


def save_example(
    data_dir: str, sid: int, oid: int, X: np.ndarray, y: np.ndarray
):
    eid = f"{sid:06}_{oid:02}"
    path = os.path.join(data_dir, f"{eid}.p")

    util.save_pickle(path=path, data=[X, y, sid, oid, path])
    return path


def sid_oid_to_pickle_path(dataset_dir: str, sid: int, oid: int):
    eid = f"{sid:06}_{oid:02}"
    path = os.path.join(dataset_dir, f"{eid}.p")
    return path


def path_to_sid_oid(path: str):
    eid = os.path.splitext(os.path.basename(path))[0]
    sid, oid = eid.split("_")
    return int(sid), int(oid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states_dir",
        required=True,
        type=str,
        help="The directory to load states from.",
    )
    parser.add_argument(
        "--img_dir",
        required=True,
        type=str,
        help="The directory to load images from.",
    )
    parser.add_argument(
        "--cam_dir",
        required=False,
        type=str,
        help="The directory to load camera parameters from, if args.coordinate_frame is 'camera'.",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        type=str,
        help="The destination directory to save the data in.",
    )
    parser.add_argument(
        "--start_sid",
        required=True,
        type=int,
        help="The start scene ID to include in the dataset.",
    )
    parser.add_argument(
        "--end_sid",
        required=True,
        type=int,
        help="The end scene ID to include in the dataset.",
    )
    parser.add_argument(
        "--objects_to_include",
        required=True,
        type=int,
        help="The first N objects to include in the dataset.",
    )
    parser.add_argument(
        "--camera_control",
        required=True,
        type=str,
        choices=["all", "center", "stack"],
        help="The method of controlling the camera.",
    )
    parser.add_argument(
        "--coordinate_frame",
        required=True,
        type=str,
        choices=["world", "camera", "unity_camera"],
        help="The coordinate frame to generate labels in.",
    )
    args = parser.parse_args()
    main(args)
