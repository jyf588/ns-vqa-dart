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

from ns_vqa_dart.bullet import dash_object, seg, util


def main(args: argparse.Namespace):
    util.delete_and_create_dir(dir=args.dst_dir)
    if not args.disable_pngs:
        util.delete_and_create_dir(dir=args.png_dir)
    # os.makedirs(args.dst_dir, exist_ok=True)

    if args.format == "flat":
        src_examples = []
        for f in sorted(os.listdir(args.states_dir)):
            sid = f.split(".")[0]
            state_path = os.path.join(args.states_dir, f)
            e = (sid, state_path)
            src_examples.append(e)
    elif args.format == "nested":
        src_examples = []
        for t in sorted(os.listdir(args.states_dir)):
            t_dir = os.path.join(args.states_dir, t)
            for f in sorted(os.listdir(t_dir)):
                state_path = os.path.join(t_dir, f)
                frame_id = f.split(".")[0]
                sid = f"{t}_{frame_id}"
                e = (sid, state_path)
                src_examples.append((e))

    paths = []
    # Loop over the scene IDs.
    for idx in tqdm(range(args.start_sid, args.end_sid)):
        sid, state_path = src_examples[idx]
        # Load the state for the current scene ID.
        state = util.load_pickle(path=state_path)

        oids = [oid for oid in range(len(state["objects"]))]

        # Only generate for the first two objects if the camera control is
        # stacking. Here we are assuming that the first two objects in the
        # state dictionary always corresponds to the two objects in the stack.
        oids_to_include = oids
        if args.objects_to_include == 2:
            oids_to_include = oids[:2]
        elif args.objects_to_include == 1:
            oids_to_include = oids[:1]
        elif args.objects_to_include == -1:
            pass
        else:
            raise ValueError(
                f"Invalid number of objects to include: {args.objects_to_include}"
            )

        X_list, y_list, oid_list = load_X_and_y(
            sid=sid,
            state=state,
            img_dir=args.img_dir,
            cam_dir=args.cam_dir,
            coordinate_frame=args.coordinate_frame,
        )

        # Save the input data.
        for X, y, oid in zip(X_list, y_list, oid_list):
            # Skip over the example if there are issues with the example (e.g.,
            # object is completely occluded)
            if X is None and y is None:
                print(f"Skipping occluded example sid: {sid}\toid: {oid}")
                continue

            if oid not in oids_to_include:
                continue

            # Save the pickle file.
            path = save_example(data_dir=args.dst_dir, sid=sid, oid=oid, X=X, y=y)
            paths.append(path)

            # Save the png.
            if not args.disable_pngs:
                input_img_rgb = np.hstack([X[:, :, :3], X[:, :, 3:6]])
                eid = f"{sid}_{oid:02}"
                path = os.path.join(args.png_dir, f"{eid}.png")
                imageio.imwrite(path, input_img_rgb)

    # Save a partition of the data.
    # split_id = int(len(paths) * 0.8)
    # partition = {
    #     "train": paths[:split_id],
    #     "val": paths[split_id:],
    #     "test": paths[split_id:],
    # }
    # util.save_json(path=os.path.join(args.dst_dir, "partition.json"), data=partition)
    # train = len(partition["train"])
    # val = len(partition["val"])
    # test = len(partition["test"])
    # print(f"Saved partition. Train: {train}\tValidation: {val}\tTest: {test}")


def load_X_and_y(
    sid: int, state: Dict, img_dir: str, cam_dir: str, coordinate_frame: str,
):

    # Load the image and segmentation for the object.
    rgb, masks, oid_list = load_rgb_and_seg(img_dir=img_dir, sid=sid)

    X_list, y_list = [], []
    for mask, oid in zip(masks, oid_list):
        # TODO: Exclude example if mask area is zero.
        X = dash_object.compute_X(img=rgb, mask=mask, keep_occluded=False)

        # The object is completely occluded so we throw the example out.
        if X is None:
            return None, None

        # Prepare camera parameters if we are using camera coordinate
        # frame.
        if coordinate_frame in ["camera", "unity_camera"]:
            cam_position, cam_orientation = load_camera_pose(cam_dir=cam_dir, sid=sid)
        else:
            raise ValueError(f"Invalid coordinate frame: {coordinate_frame}")
        y = dash_object.compute_y(
            odict=state["objects"][oid],
            coordinate_frame=coordinate_frame,
            cam_position=cam_position,
            cam_orientation=cam_orientation,
        )
        X_list.append(X)
        y_list.append(y)
    return X_list, y_list, oid_list


def load_rgb_and_seg(img_dir: str, sid: int) -> Tuple[np.ndarray, np.ndarray]:
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
    rgb, seg_img = load_rgb_and_seg_img(img_dir=img_dir, sid=sid)

    # Convert the segmentation image into an array.
    # TODO: Do this before saving the segmentation.
    masks, oids = seg.seg_img_to_map(seg_img)
    return rgb, masks, oids


def load_rgb_and_seg_img(img_dir: str, sid: int) -> Tuple[np.ndarray, np.ndarray]:
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
    rgb_path = os.path.join(img_dir, "rgb", f"{sid}_0.png")
    seg_path = os.path.join(img_dir, "seg", f"{sid}_0.png")
    rgb = imageio.imread(uri=rgb_path)
    seg_img = imageio.imread(uri=seg_path)
    return rgb, seg_img


def load_third_person_image(img_dir: str, sid: str):
    path = os.path.join(img_dir, "third/rgb", f"{sid:06}_.png")
    img = imageio.imread(path)
    return img


def load_camera_pose(cam_dir: str, sid: int):
    """Creates a camera with same parameters as camera used to capture images
    for the specified object in the specified scene.

    Args:
        cam_dir: The directory containing camera JSON files.
        sid: The sensor ID.
        oid: The object ID.
    
    Returns:
        cam: A BulletCamera.
    """
    path = os.path.join(cam_dir, f"{sid}.json")
    params = util.load_json(path=path)["0"]
    position = params["camera_position"]
    orientation = params["camera_orientation"]
    return position, orientation


def save_example(data_dir: str, sid: int, oid: int, X: np.ndarray, y: np.ndarray):
    eid = f"{sid}_{oid:02}"
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
        "--format", required=True, type=str, help="The format.",
    )
    parser.add_argument(
        "--states_dir",
        required=True,
        type=str,
        help="The directory to load states from.",
    )
    parser.add_argument(
        "--img_dir", required=True, type=str, help="The directory to load images from.",
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
        "--png_dir", type=str, help="The destination directory to save the data in.",
    )
    parser.add_argument(
        "--disable_pngs", action="store_true", help="Don't save pngs..",
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
        "--coordinate_frame",
        required=True,
        type=str,
        choices=["world", "camera", "unity_camera"],
        help="The coordinate frame to generate labels in.",
    )
    args = parser.parse_args()
    main(args)
