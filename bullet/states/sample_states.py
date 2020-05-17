"""Subsamples states.

The expected source file structure is:
    <src_dir>/
        <sid>.p

The output generated file structure will be:
    <dst_dir>/
        <sid>.p
"""
import os
import sys
import pprint
import random
import shutil
import argparse
import collections
from tqdm import tqdm

from ns_vqa_dart.bullet import util


def main(args: argparse.Namespace):
    # Set seed since we use a random generator.
    random.seed(args.seed)

    # Verify that the destination directory does not already exist.
    util.delete_and_create_dir(dir=args.dst_dir)

    # Collect examples.
    examples = []
    for t_int in range(args.start_trial, args.end_trial):
        t = f"{t_int:06}"
        trial_dir = os.path.join(args.src_dir, t)
        for fname in sorted(os.listdir(trial_dir)):
            path = os.path.join(trial_dir, fname)
            assert os.path.exists(path)
            examples.append((t, fname))

    n = len(examples)
    print(f"Number of states to sample from: {n}")

    # Verify that the sample size is less than or equal to the number of source files.
    assert args.sample_size <= n

    # Sample indices.
    sorted_sample_idxs = sorted(random.sample(range(n), args.sample_size))

    # Select the sampled idxs.
    sampled_examples = []
    t2count = collections.Counter()
    for idx in sorted_sample_idxs:
        sampled_e = examples[idx]
        sampled_examples.append(sampled_e)
        t2count[sampled_e[0]] += 1

    # Print stats.
    sum_over_trials = sum(list(t2count.values()))
    n_trials = len(t2count)
    print()
    print(f"***Sampling Stats:***")
    print(f"Number of sampled examples: {len(sampled_examples)}")
    print(f"Number of trials sampled from: {n_trials}")
    print(f"Sum of examples over trials: {sum_over_trials}")
    print(f"Average number of frames sampled per trial: {sum_over_trials / n_trials}")
    print(
        f"***Note: Currently, we might not sample from all trials since we sample "
        f"over all trials, not per trial. Will be fixed in the next iteration. We need "
        f"to sample this way first in order to reproduce old code."
    )
    print()

    # Create trial directories in the destination folder.
    for t in t2count.keys():
        dst_t_dir = os.path.join(args.dst_dir, t)
        util.delete_and_create_dir(dir=dst_t_dir)

    # Copy the sampled files into the destination directory.
    print(f"Copying sampled files from: {args.src_dir}")
    print(f"Into: {args.dst_dir}")
    print(f"Printing the first five examples:")
    for idx, (t, fname) in enumerate(sampled_examples):
        src_path = os.path.join(args.src_dir, t, fname)
        dst_path = os.path.join(args.dst_dir, t, fname)
        shutil.copy(src_path, dst_path)
        if idx < 5:
            print(
                f"Copying file to and from:\n"
                f"\tSrc: {src_path}\n"
                f"\tDst: {dst_path}"
            )


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
        "--start_trial",
        required=True,
        type=int,
        help="The trial to start sampling from.",
    )
    parser.add_argument(
        "--end_trial",
        required=True,
        type=int,
        help="The trial to end sampling at (exclusive).",
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
