import argparse
import copy
import numpy as np
import os
import pickle
from tqdm import tqdm
from typing import *

from ns_vqa_dart.bullet.dash_dataset import DashDataset
from ns_vqa_dart.bullet import dash_object
from ns_vqa_dart.bullet import gen_dataset
from ns_vqa_dart.bullet import util

AXIS_NAMES = ["x", "y", "z"]
MULTIPLIER = {"radius": 100, "height": 100, "position": 100, "height": 100}
UNITS = {
    "radius": "cm",
    "height": "cm",
    "position": "cm",
    "up_vector": "L1",
    "height": "cm",
}


def main(args: argparse.Namespace):
    pred_dicts = util.load_json(path=args.pred_path)
    compute_metrics(
        pred_dicts=pred_dicts, coordinate_frame=args.coordinate_frame
    )


def compute_metrics(cam_dir: str, sid2info: List[Dict], coordinate_frame: str):
    cls_correct = {k: 0 for k in ["shape", "color"]}
    errors = {
        "radius": 0.0,
        "height": 0.0,
        "position": np.zeros((3,)),
        "up_vector": np.zeros((3,)),
    }
    reg_error = {
        "pos": copy.deepcopy(errors),
        "neg": copy.deepcopy(errors),
        "abs": copy.deepcopy(errors),
    }
    n_total = 0

    for sid in sid2info.keys():
        for oid, info in sid2info[sid].items():
            sid, oid = int(sid), int(oid)
            pred = info["pred"]
            labels = info["labels"]

            # Convert from vectors to dictionaries.
            camera = gen_dataset.create_camera(
                cam_dir=cam_dir, sid=sid, oid=oid
            )
            gt_y_dict = dash_object.y_vec_to_dict(
                y=labels, coordinate_frame=coordinate_frame, camera=camera
            )
            pred_y_dict = dash_object.y_vec_to_dict(
                y=pred, coordinate_frame=coordinate_frame, camera=camera
            )

            for k in cls_correct.keys():
                if gt_y_dict[k] == pred_y_dict[k]:
                    cls_correct[k] += 1

            for k in errors.keys():
                diff = np.array(pred_y_dict[k]) - np.array(gt_y_dict[k])
                pos_diff = diff.copy()
                neg_diff = diff.copy()

                if type(diff) == np.float64:
                    pos_diff = diff if diff >= 0.0 else 0.0
                    neg_diff = diff if diff <= 0.0 else 0.0
                elif type(diff) == np.ndarray:
                    pos_diff[pos_diff < 0] = 0.0
                    neg_diff[neg_diff > 0] = 0.0
                else:
                    raise ValueError(f"Invalid type: {type(diff)}")

                l1 = np.abs(diff)
                reg_error["pos"][k] += pos_diff
                reg_error["neg"][k] += neg_diff
                reg_error["abs"][k] += l1

            n_total += 1

    np.set_printoptions(2)

    print(f"Classification Accuracies:")
    for k, v in cls_correct.items():
        print(f"\t{k}: {v / n_total * 100:.2f} ({v}/{n_total})")

    print(f"Regression Errors:")
    for k in errors.keys():
        for err_type in reg_error.keys():
            v = reg_error[err_type][k]

            # Multiply by multiplier for the key if specified.
            if k in MULTIPLIER:
                v *= MULTIPLIER[k]
            units = UNITS[k]

            # Print out the results.
            if type(v) in [np.float64, float]:
                print(
                    f"\t{k} {err_type} ({units}): {v / n_total:.2f} ({v:.2f}/{n_total})"
                )
            elif type(v) == np.ndarray:
                print(f"\t{k} {err_type} ({units}):")
                for axis_i, v_i in enumerate(v):
                    print(
                        f"\t\t{AXIS_NAMES[axis_i]}: {v_i / n_total:.2f} ({v_i:.2f}/{n_total})"
                    )
            else:
                raise ValueError(f"Unrecognized type: {type(v)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory containing the dataset.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="The path to the predictions JSON.",
    )
    parser.add_argument(
        "--coordinate_frame",
        type=str,
        required=True,
        choices=["world", "camera"],
        help="The coordinate frame that predictions are in.",
    )
    args = parser.parse_args()
    main(args)
