"""Subsamples states.

The expected source file structure is:
    <src_dir>/
        <sid>.p

The output generated file structure will be:
    <dst_dir>/
        <sid>.p
"""
import argparse
import os
import pprint
import random
import shutil
import sys


def main(args: argparse.Namespace):
    random.seed(args.seed)

    src_fnames = os.listdir(args.src_dir)
    n_src = len(src_fnames)

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
    sampled_fnames = [sorted_fnames[idx] for idx in sorted_sample_idxs]
    print(f"sampled_fnames: {len(sampled_fnames)}")

    # Copy the sampled files into the destination directory.
    if os.path.exists(args.dst_dir):
        user_input = input(
            f"dst dir already exists: {args.dst_dir}. Delete and continue? [Y/n]"
        )
        if user_input == "Y":
            shutil.rmtree(args.dst_dir)
        else:
            print(f"user_input: {user_input}. Exiting.")
            sys.exit(0)
    os.makedirs(args.dst_dir)
    for fname in sampled_fnames:
        src_path = os.path.join(args.src_dir, fname)
        dst_path = os.path.join(args.dst_dir, fname)
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
