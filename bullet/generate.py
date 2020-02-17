"""
Generates a dataset containing various scenes of random objects placed randomly
on a tabletop, all upright. Currently does not generate examples of stacked 
objects. Robot arm is not in view for this dataset.
"""
import argparse
import time

from bullet.scenes import RandomSceneGenerator


def main(args: argparse.Namespace):
    # gui: 4 hours
    # direct: 1 hour

    generator = RandomSceneGenerator(
        seed=1,
        render_mode="direct",
        egocentric=True,
        dataset_dir=args.dataset_dir,
        n_objs_bounds=(3, 7),
        obj_dist_thresh=0.25,
        max_retries=50,
        shapes=["box", "cylinder"],
        sizes=["small", "large"],
        colors=["red", "blue", "yellow", "green"],
        x_bounds=(0.0, 0.4),
        y_bounds=(-0.5, 0.5),
        z_bounds=(0.0, 0.0),
        roll_bounds=(-5.0, 5.0),
        tilt_bounds=(35.0, 55.0),
        pan_bounds=(-5.0, 5.0),
        degrees=True,
    )
    generator.generate_scenes(n=args.n_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory to store the dataset in.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        required=True,
        help="The number of examples to generate.",
    )
    args = parser.parse_args()
    main(args)
