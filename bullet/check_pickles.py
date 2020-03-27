"""
Checks whether there are any corrupt pickle files in a dataset.
"""
import argparse
import os
from tqdm import tqdm

from ns_vqa_dart.bullet import util


def main(args: argparse.Namespace):
    for fname in tqdm(os.listdir(args.pickles_dir)):
        if not fname.endswith(".p"):
            continue
        path = os.path.join(args.pickles_dir, fname)
        try:
            util.load_pickle(path=path)
        except EOFError as e:
            print(e)
            print(f"EOFError detected for: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pickles_dir",
        type=str,
        required=True,
        help="The directory containing pickle files.",
    )
    args = parser.parse_args()
    main(args=args)
