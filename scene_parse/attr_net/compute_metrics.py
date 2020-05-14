import os
import sys
import numpy as np

from ns_vqa_dart.bullet import util, dash_object, metrics


def main():
    run_dir = "/home/mguo/outputs/placing_v003_2K_20K/2020_04_22_04_35/eval/2020_05_13_23_53_42"
    load_path = os.path.join(run_dir, "preds.p")

    predictions = util.load_pickle(load_path)
    train_m = metrics.Metrics()
    val_m = metrics.Metrics()

    for sid, pred_dict in predictions.items():
        set = "train"
        if sid >= 16000:
            set = "val"
        yhat = np.array(pred_dict["pred"])
        y = np.array(pred_dict["y"])

        pred_dict = dash_object.y_vec_to_dict(y=yhat, coordinate_frame="world")
        gt_dict = dash_object.y_vec_to_dict(y=y, coordinate_frame="world")
        if set == "train":
            train_m.add_example(gt_dict=gt_dict, pred_dict=pred_dict)
        elif set == "val":
            val_m.add_example(gt_dict=gt_dict, pred_dict=pred_dict)

    train_save_path = os.path.join(run_dir, "train_metrics.txt")
    val_save_path = os.path.join(run_dir, "val_metrics.txt")

    print(f"Saving metrics to: {train_save_path}")
    print(f"Saving metrics to: {val_save_path}")

    assert not os.path.exists(train_save_path)
    assert not os.path.exists(val_save_path)

    sys.stdout = open(train_save_path, "wt")
    train_m.print()

    sys.stdout = open(val_save_path, "wt")
    val_m.print()


if __name__ == "__main__":
    main()
