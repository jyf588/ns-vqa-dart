"""Subsamples states.

The expected source file structure is:
    <src_dir>/
        <sid>.p

The output generated file structure will be:
    <dst_dir>/
        <sid>.p
"""
import argparse
import collections
import os
import pprint
import random
import shutil
import sys

from ns_vqa_dart.bullet import util


def main(args: argparse.Namespace):
    random.seed(args.seed)

    src_fnames = os.listdir(args.src_dir)
    n_src = len(src_fnames)
    print(f"Number of source frames: {n_src}")

    # Verify that the sample size is less than or equal to the number of
    # source files.
    assert args.sample_size <= n_src

    # Collect the source paths.
    sorted_fnames = []
    for fname in sorted(src_fnames):
        sorted_fnames.append(fname)

    # Sample indices.
    sorted_sample_idxs = sorted(random.sample(range(n_src), args.sample_size))
    print(f"sorted_sample_idxs: {len(sorted_sample_idxs)}")

    # Select the sampled idxs.
    sampled_fnames = []
    trial_idx2count = collections.Counter()
    for idx in sorted_sample_idxs:
        sampled_fnames.append(sorted_fnames[idx])
        trial_idx = idx // 100
        trial_idx2count[trial_idx] += 1
    print(f"sampled_fnames: {len(sampled_fnames)}")
    avg_per_trial = sum(list(trial_idx2count.values())) / len(trial_idx2count)
    print(f"Average number of frames sampled per trial: {avg_per_trial}")

    # Copy the sampled files into the destination directory.
    util.delete_and_create_dir(dir=args.dst_dir)
    for dst_sid, fname in enumerate(sampled_fnames):
        src_path = os.path.join(args.src_dir, fname)
        dst_path = os.path.join(args.dst_dir, f"{dst_sid:06}.p")
        shutil.copy(src_path, dst_path)
    print(f"Number of sampled files: {len(sampled_fnames)}")


if __name__ == "__main__":
    print(f"*****sample_states.py*****")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", required=True, type=int, help="The random seed.",
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        type=str,
        help="The source directory to sample states from.",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        type=str,
        help="The destination directory to generate states to.",
    )
    parser.add_argument(
        "--sample_size",
        required=True,
        type=int,
        help="The number of states to sample. Must be less than or equal to the number of states in the source directory.",
    )
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args=args)
