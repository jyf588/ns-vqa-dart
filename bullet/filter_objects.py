import argparse
from tqdm import tqdm

from bullet.dash_dataset import DashDataset


def main(args: argparse.Namespace):
    dataset = DashDataset(args.dataset_dir)
    filtered_dataset = DashDataset(args.filtered_dataset_dir)

    assert args.dataset_dir != args.filtered_dataset_dir

    n_removed_objects, n_empty_scenes = 0, 0

    img_ids = dataset.load_example_ids()
    for img_id in tqdm(img_ids):
        objects, camera, rgb, mask = dataset.load_example(eid=img_id)
        filtered_objects = dataset.filter_out_of_view_objects(objects=objects)
        filtered_dataset.save_example(
            objects=filtered_objects, camera=camera, rgb=rgb, mask=mask
        )

        n_removed_objects = len(objects) - len(filtered_objects)
        if len(filtered_objects) == 0:
            n_empty_scenes += 1

    print(f"Finished filtering.")
    print(f"Number of removed objects: {n_removed_objects}")
    print(f"Number of scenes with zero objects: {n_empty_scenes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory containing the original, unfiltered dataset.",
    )
    parser.add_argument(
        "--filtered_dataset_dir",
        type=str,
        required=True,
        help="The directory to save the new filtered dataset in.",
    )
    args = parser.parse_args()
    main(args)
