import os
import sys
import argparse
import numpy as np

from ns_vqa_dart.bullet import util, dash_object
from ns_vqa_dart.bullet.metrics import Metrics


def main(args):
    eval_dir = os.path.join(args.run_dir, "eval", args.eval_name)

    split2metrics = {}
    split2load_path = {}
    split2save_path = {}
    for split in ["train", "val"]:
        split2metrics[split] = Metrics()
        split2load_path[split] = os.path.join(eval_dir, f"{split}_preds.p")
        save_path = os.path.join(eval_dir, f"{split}_metrics.txt")
        split2save_path[split] = save_path
        assert not os.path.exists(save_path)

    for split, load_path in split2load_path.items():
        predictions = util.load_pickle(load_path)

        for sid, pred_dict in predictions.items():
            yhat = np.array(pred_dict["pred"])
            y = np.array(pred_dict["y"])

            pred_dict = dash_object.y_vec_to_dict(y=yhat, coordinate_frame="world")
            gt_dict = dash_object.y_vec_to_dict(y=y, coordinate_frame="world")
            split2metrics[split].add_example(gt_dict=gt_dict, pred_dict=pred_dict)

    for split, metrics in split2metrics.items():
        save_path = split2save_path[split]
        sys.stdout = open(save_path, "wt")
        metrics.print()
        print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--eval_name", required=True, type=str)
    args = parser.parse_args()
    main(args)
