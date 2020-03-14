"""Generates a stacking dataset from saved states from running enjoy.py."""
import os
import random

from .camera import BulletCamera


def main():
    seed = 0
    random.seed(seed)

    # Paths
    dataset_dir = "/home/michelle/datasets/stacking_v003"
    state_dirs = [
        "/home/michelle/datasets/delay_box_states",
        "/home/michelle/datasets/delay_cyl_states",
    ]
    n_examples_per_state = 10

    # Camera
    camera = BulletCamera(
        p=p,
        position=[-0.2237938867122504, 0.03198004185028341, 0.5425],
        rotation=[0.0, 50.0, 0.0],
        offset=[0.0, 0.0, 0.0],
    )

    dataset = DashDataset(dataset_dir=dataset_dir)

    dataset_paths = []
    for state_name in state_names:
        states_dir = os.path.join(datasets_dir, state_name)

        paths = []
        for fname in os.listdir(states_dir):
            path = os.path.join(states_dir, fname)
            paths.append(path)

        # Sample a subset.
        random.shuffle(paths)
        paths = paths[:n_examples_per_state]

        dataset_paths += paths

    for path in dataset_paths:
        objects = []
        dataset.save_example(objects=objects, camera=camera)


if __name__ == "__main__":
    main()
