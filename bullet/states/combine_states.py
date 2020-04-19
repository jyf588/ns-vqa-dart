"""
Combines an arbitrary number of state sets into a single set of states.

Expected format of source state directories:
    <src_dir>/
        <sid>.p

Generated format of destination state directory:
    <dst_dir>/
        <sid>.p
    where <sid>'s are tightly ordered from 0 to <args.n_states> - 1.
"""
import argparse
import os
import pprint
import random
import shutil
from tqdm import tqdm
from typing import *

import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    # Set the random seed.
    random.seed(args.seed)

    # Create the destination directory.
    os.makedirs(args.dst_dir, exist_ok=True)

    # Load the partition JSON file.
    partition_dict = util.load_json(path=args.partition_path)

    print("Loaded partition JSON file:")
    pprint.pprint(partition_dict)
    print()

    set2frac = partition_dict["sets"]
    split2frac = partition_dict["splits"]

    # Verify that the proportions add up to 1.
    assert 1.0 == sum([p for p in set2frac.values()])
    assert 1.0 == sum([p for p in split2frac.values()])

    """
    Subsample frames from each set, without shuffling.
    This should yield:
        planning: 25% of 20K = 5K frames
        placing: 75% of 20K = 15K frames
    Now, splitting into train and validation sets:
        planning:
            train: 80% of 5K = 4K frames
            val: 20% of 5K = 1K frames
        placing:
            train: 80% of 15K = 12K frames
            val: 20% of 15K = 3K frames
    0-4K: planning train
    4K-16K: placing train
    16K-17K: planning val
    17K-20K: placing val
    """

    # Sample frames from each of the sets.
    src_train_paths = []
    src_val_paths = []
    for set_dir, set_frac in set2frac.items():
        # Compute the total number of states in the source directory.
        paths = [
            os.path.join(set_dir, fname)
            for fname in sorted(os.listdir(set_dir))
        ]
        n_total = len(paths)

        # Compute the number of states to sample from the current set.
        sample_size = int(args.n_states * set_frac)

        # Subsample indices, and sort them.
        sorted_sample_idxs = sorted(random.sample(range(n_total), sample_size))

        # Collect the state filepaths for the selected indices.
        sorted_sample_paths = [paths[idx] for idx in sorted_sample_idxs]

        # Split the sampled paths into training and validation sets.
        n_train = int(sample_size * split2frac["train"])
        n_val = int(sample_size * split2frac["val"])

        sorted_train_paths = sorted_sample_paths[:n_train]
        sorted_val_paths = sorted_sample_paths[n_train:]
        assert len(sorted_val_paths) == n_val

        src_train_paths += sorted_train_paths
        src_val_paths += sorted_val_paths

        print(f"***Set Info***")
        print(f"Set dir: {set_dir}")
        print(f"Sample size: {sample_size}")
        print(f"Num train: {len(sorted_train_paths)}")
        print(f"Num val: {len(sorted_val_paths)}")
        print()

    # Combine all dataset paths into a single list.
    dataset_paths = src_train_paths + src_val_paths

    dataset_info = {
        "src_train_paths": src_train_paths,
        "src_val_paths": src_val_paths,
        "src2dst": {},
    }

    # Copy source files to the destination directory.
    print(f"Copying selected states to {args.dst_dir}...")

    n_total_objects = 0
    for path_i, src_path in enumerate(tqdm(dataset_paths)):
        state = util.load_pickle(path=src_path)
        n_total_objects += len(state["objects"])

        dst_path = os.path.join(args.dst_dir, f"{path_i:06}.p")
        shutil.copyfile(src_path, dst_path)

        # Store the mapping from source to destination path.
        dataset_info["src2dst"][src_path] = dst_path

    print("*****Stats:*****")
    print(f"Number of states: {len(dataset_paths)}")
    print(f"Number of objects: {n_total_objects}")
    print(f"Num train: {len(src_train_paths)}")
    print(f"Num val: {len(src_val_paths)}")

    # Save dataset information.
    util.save_json(
        os.path.join(args.dst_dir, "dataset_info.json"), dataset_info
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=1, help="The random seed to use.",
    )
    parser.add_argument(
        "--partition_path",
        required=True,
        type=str,
        help="The path to the JSON file specifying input directories and their relative proportions to be included in the combined set.",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        help="The destination directory to save the final set of states.",
        required=True,
    )
    parser.add_argument(
        "--n_states",
        type=int,
        help="The number of states to include in the final set of states.",
        required=True,
    )
    args = parser.parse_args()
    print(f"Arguments:")
    pprint.pprint(vars(args))
    main(args=args)
