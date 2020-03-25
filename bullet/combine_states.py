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
import random
import shutil
from typing import *


def main(args: argparse.Namespace):
    # Compute the number of examples to sample from each source directory in
    # order to achieve the user-specified total number of examples in the
    # final set of states.
    n_src_sets = len(args.src_dirs)
    n_states_per_set = args.n_states // n_src_sets
    idx2path = {}
    start_idx = 0
    for dir_i, src_dir in enumerate(args.src_dirs):
        # If it's the last partition, grab everything until the end.
        if dir_i == n_src_sets - 1:
            end_idx = args.n_states
        else:
            end_idx = start_idx + n_states_per_set

        # Subsample paths from the current directory.
        n_examples = end_idx - start_idx
        paths = sample_states(src_dir=src_dir, n_samples=n_examples)

        # Assign paths of indexes in the final set of states.
        for i, idx in enumerate(range(start_idx, end_idx)):
            idx2path[idx] = paths[i]

        print(f"Source dir: {src_dir}")
        print(f"\tRange: [{start_idx}, {end_idx})")
        print(f"\tTotal examples: {n_examples}")

        start_idx = end_idx

    # Copy source files to the destination directory.
    os.makedirs(args.dst_dir, exist_ok=True)
    for idx, src_path in idx2path.items():
        dst_path = os.path.join(args.dst_dir, f"{idx:06}.p")
        print(src_path)
        # print(dst_path)
        shutil.copyfile(src_path, dst_path)


def sample_states(src_dir: str, n_samples: int) -> List[str]:
    """Samples `n_samples` states from a states directory.

    Args:
        src_dir: The source directory of states to sample from.
        n_samples: The number of samples to select.

    Returns:
        filepaths: A list of sampled filepaths.
    """
    fnames = os.listdir(src_dir)

    # Sample without replacement, in sorted order.
    fnames = sorted(random.sample(fnames, n_samples))
    filepaths = []
    for fname in fnames:
        filepaths.append(os.path.join(src_dir, fname))
    return filepaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dirs",
        nargs="+",
        help="A list of source state directories to combine",
        required=True,
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
    main(args=args)
