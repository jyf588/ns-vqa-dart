import argparse
import numpy as np
from tqdm import tqdm

from bullet.dash_dataset import DashDataset
import bullet.dash_object
import bullet.util

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

    dataset = DashDataset(args.dataset_dir)
    pred_dicts = bullet.util.load_json(path=args.pred_path)

    cls_correct = {k: 0 for k in ["shape", "color"]}
    reg_error = {
        "radius": 0,
        "height": 0,
        "position": np.zeros((3,)),
        "up_vector": np.zeros((3,)),
        "height": 0,
    }
    n_total = 0

    for pred_dict in tqdm(pred_dicts):
        o = dataset.load_object_for_img_id_and_oid(
            img_id=pred_dict["img_id"], oid=pred_dict["oid"]
        )
        odict = o.to_json()

        camera = dataset.load_camera_for_eid(eid=pred_dict["img_id"])
        y_dict = bullet.dash_object.y_vec_to_dict(
            y=pred_dict["pred"],
            coordinate_frame=args.coordinate_frame,
            camera=camera,
        )

        for k in cls_correct.keys():
            if odict[k] == y_dict[k]:
                cls_correct[k] += 1

        for k in reg_error.keys():
            l1 = np.abs(np.array(odict[k]) - np.array(y_dict[k]))
            reg_error[k] += l1

        n_total += 1

    np.set_printoptions(2)

    print(f"Classification Accuracies:")
    for k, v in cls_correct.items():
        print(f"\t{k}: {v / n_total * 100:.2f} ({v}/{n_total})")

    print(f"Regression Errors:")
    for k, v in reg_error.items():
        if k in MULTIPLIER:
            v *= MULTIPLIER[k]
        units = UNITS[k]
        if type(v) == np.float64:
            print(f"\t{k} ({units}): {v / n_total:.2f} ({v:.2f}/{n_total})")
        elif type(v) == np.ndarray:
            print(f"\t{k} ({units}):")
            for axis_i, v_i in enumerate(v):
                print(
                    f"\t\t{AXIS_NAMES[axis_i]}: {v_i / n_total:.2f} ({v_i:.2f}/{n_total})"
                )


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
