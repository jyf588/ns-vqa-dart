import argparse
import numpy as np

from bullet.dash_dataset import DashDataset
import bullet.dash_object
import bullet.util


def main(args: argparse.Namespace):

    dataset = DashDataset(args.dataset_dir)
    pred_dicts = bullet.util.load_json(path=args.pred_path)

    cls_correct = {k: 0 for k in ["shape", "size", "color"]}
    reg_error = {
        "position": np.zeros((3,)),
        "up_vector": np.zeros((3,)),
        "height": 0,
    }
    n_total = 0

    for pred_dict in pred_dicts:
        y_dict = bullet.dash_object.y_vec_to_dict(y=pred_dict["pred"])
        o = dataset.load_object(
            img_id=pred_dict["img_id"], oid=pred_dict["oid"]
        )
        odict = o.to_json()

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
        if type(v) == np.float64:
            print(f"\t{k}: {v_i / n_total:.2f} ({v:.2f}/{n_total})")
        elif type(v) == np.ndarray:
            print(f"\t{k}:")
            for axis_i, v_i in enumerate(v):
                print(
                    f"\t\taxis {axis_i}: {v_i / n_total:.2f} ({v_i:.2f}/{n_total})"
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
    args = parser.parse_args()
    main(args)
