import argparse
import copy
import numpy as np
import os
import pickle
import pprint
from tqdm import tqdm
from typing import *

from ns_vqa_dart.bullet.dash_dataset import DashDataset
from ns_vqa_dart.bullet import dash_object
from ns_vqa_dart.bullet import gen_dataset
from ns_vqa_dart.bullet import util

AXIS_NAMES = ["x", "y", "z"]
MULTIPLIER = {"radius": 100, "height": 100, "position": 100}
UNITS = {
    "radius": "cm",
    "height": "cm",
    "position": "cm",
    "up_vector": "L1",
}


class Metrics:
    def __init__(self):
        # Initialize the error counters for various categories.
        self.counts_dict = {k: 0 for k in ["shape", "color"]}
        self.l1_errors_dict = {
            "radius": 0.0,
            "height": 0.0,
            "position": np.zeros((3,)),
            "up_vector": np.zeros((3,)),
        }
        self.reg_errors_dict = {
            "pos": copy.deepcopy(self.l1_errors_dict),
            "neg": copy.deepcopy(self.l1_errors_dict),
            "abs": copy.deepcopy(self.l1_errors_dict),
        }
        self.n_total = 0

    def add_example(self, gt_dict, pred_dict):
        """Computes errors, and adds them to the running total.

        Args:
            gt_dict: A dictionary containing ground truth values, with the 
                format: {
                    "shape": <str>,
                    "color": <str>,
                    "height": <float>,
                    "radius": <float>,
                    "position": <List[float]>,
                    "up_vector": <List[float]>,
                }
            pred_dict: A dictionary containing predicted values, with the same
                format as `gt_dict`.
    
        """
        for k in self.counts_dict.keys():
            if gt_dict[k] == pred_dict[k]:
                self.counts_dict[k] += 1

        for k in self.l1_errors_dict.keys():
            diff = np.array(pred_dict[k]) - np.array(gt_dict[k])
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
            self.reg_errors_dict["pos"][k] += pos_diff
            self.reg_errors_dict["neg"][k] += neg_diff
            self.reg_errors_dict["abs"][k] += l1

        self.n_total += 1

    def print(self):
        """Prints the metrics computed thus far."""
        np.set_printoptions(2)

        print(f"Classification Accuracies:")
        for k, v in self.counts_dict.items():
            print(f"\t{k}: {v / self.n_total * 100:.2f} ({v}/{self.n_total})")

        print(f"Regression Errors:")
        for k in self.l1_errors_dict.keys():
            for err_type in self.reg_errors_dict.keys():
                v = self.reg_errors_dict[err_type][k]

                # Multiply by multiplier for the key if specified.
                if k in MULTIPLIER:
                    v *= MULTIPLIER[k]
                units = UNITS[k]

                # Print out the results.
                if type(v) in [np.float64, float]:
                    print(
                        f"\t{k} {err_type} ({units}): {v / self.n_total:.2f} ({v:.2f}/{self.n_total})"
                    )
                elif type(v) == np.ndarray:
                    print(f"\t{k} {err_type} ({units}):")
                    for axis_i, v_i in enumerate(v):
                        print(
                            f"\t\t{AXIS_NAMES[axis_i]}: {v_i / self.n_total:.2f} ({v_i:.2f}/{self.n_total})"
                        )
                else:
                    raise ValueError(f"Unrecognized type: {type(v)}")


def main(args: argparse.Namespace):
    pred_dicts = util.load_json(path=args.pred_path)
    compute_metrics(
        pred_dicts=pred_dicts, coordinate_frame=args.coordinate_frame
    )


def compute_metrics(
    cam_dir: str,
    sid2info: List[Dict],
    camera_control: str,
    coordinate_frame: str,
):
    metrics = Metrics()
    for sid in sid2info.keys():
        for oid, info in sid2info[sid].items():
            sid, oid = int(sid), int(oid)
            pred = info["pred"]
            labels = info["labels"]

            # Convert from vectors to dictionaries.
            cam_position, cam_orientation = gen_dataset.load_camera_pose(
                cam_dir=cam_dir,
                sid=sid,
                oid=oid,
                camera_control=camera_control,
            )
            gt_y_dict = dash_object.y_vec_to_dict(
                y=labels,
                coordinate_frame=coordinate_frame,
                cam_position=cam_position,
                cam_orientation=cam_orientation,
            )
            pred_y_dict = dash_object.y_vec_to_dict(
                y=pred,
                coordinate_frame=coordinate_frame,
                cam_position=cam_position,
                cam_orientation=cam_orientation,
            )

            metrics.add_example(gt_dict=gt_y_dict, pred_dict=pred_y_dict)
    metrics.print()


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
