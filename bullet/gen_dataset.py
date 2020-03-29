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
        <sid>/
            test_rgb.png
            test_seg.png
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
from tqdm import tqdm
from typing import *

from ns_vqa_dart.bullet.dash_dataset import DashDataset
import ns_vqa_dart.bullet.dash_object as dash_object
import ns_vqa_dart.bullet.random_objects as random_objects
import ns_vqa_dart.bullet.util as util

RGB2ID = {
    (174, 199, 232): 0,  # blue
    (152, 223, 138): 1,  # green
    (255, 152, 150): 2,  # red
    (197, 176, 213): 3,  # purple
    (219, 219, 141): 4,  # lime green
    (196, 156, 148): 5,  # brown
    (255, 187, 120): 6,  # orange
    (247, 182, 210): 7,  # pink
}


def main(args: argparse.Namespace):
    os.makedirs(args.dst_dir, exist_ok=True)

    paths = []
    # Loop over the scene IDs.
    for sid in tqdm(range(args.start_sid, args.end_sid)):
        # Load the state for the current scene ID.
        state = util.load_pickle(
            path=os.path.join(args.states_dir, f"{sid:06}.p")
        )

        # Load the image and segmentation of the scene.
        # TODO: Change to converting from png to numpy for segmentation.
        rgb, seg = load_rgb_and_seg(img_dir=args.img_dir, sid=sid)

        # Save the input data.
        for oid, odict in state["objects"].items():
            # TODO: Exclude example if mask area is zero.
            X = dash_object.compute_X(oid=oid, img=rgb, seg=seg)
            y = dash_object.compute_y(odict=odict, coordinate_frame="world")
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


def load_rgb_and_seg(img_dir: str, sid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the RGB image and segmentation map for a given scene.

    Args:
        img_dir: The directory containing the images.
        sid: The scene ID.

    Returns:
        rgb: The RGB image, of shape (H, W, 3)
        segmentation: A 2D segmentation map where each pixel stores the object 
            ID it belongs to.
    """
    rgb_path = os.path.join(img_dir, "img", f"{sid:06}.png")
    seg_path = os.path.join(img_dir, "id", f"{sid:06}.png")
    rgb = imageio.imread(uri=rgb_path)
    seg_img = imageio.imread(uri=seg_path)

    # Convert the segmentation image into an array.
    # TODO: Do this before saving the segmentation.
    H, W, _ = seg_img.shape
    seg = np.full((H, W), -1, dtype=np.uint8)
    for rgb_value, oid in RGB2ID.items():
        idxs = np.where(
            np.logical_and(
                seg_img[:, :, 0] == rgb_value[0],
                np.logical_and(
                    seg_img[:, :, 1] == rgb_value[1],
                    seg_img[:, :, 2] == rgb_value[2],
                ),
            )
        )
        seg[idxs] = oid
    return rgb, seg


def load_rgb(dataset_dir: str, sid: int):
    rgb_square = imageio.imread(
        os.path.join(dataset_dir, "rgb", f"{sid:06}.png")
    )
    return rgb_square


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
    args = parser.parse_args()
    main(args)
