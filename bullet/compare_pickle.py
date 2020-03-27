import os
import time
from tqdm import tqdm

from ns_vqa_dart.bullet import util


def main():
    pickle_dir1 = (
        "/media/michelle/68B62784B62751BC/data/datasets/dash_v002/data"
    )
    pickle_dir2 = "/home/michelle/test_pickle"

    pickle1_times = 0
    pickle2_times = 0
    n = 0

    for fname in tqdm(sorted(os.listdir(pickle_dir2))):
        pickle1 = os.path.join(pickle_dir1, fname)
        pickle2 = os.path.join(pickle_dir2, fname)

        start = time.time()
        util.load_pickle(path=pickle1)
        pickle1_times += time.time() - start

        start = time.time()
        util.load_pickle(path=pickle2)
        pickle2_times += time.time() - start

        n += 1

    print(f"pickle1: {pickle1_times / n}")
    print(f"pickle2: {pickle2_times / n}")
    print(f"n={n}")


if __name__ == "__main__":
    main()
