"""
Completes placing states generated from policy evaluation by randomly 
assigning values to attributes that are missing from the states.

Expected source directory format:
    <args.src_dir>/
        <sid>.p

Expected source format for existing <sid>.p files:
    {
        "objects": [
            {
                "shape": <shape>,
                "radius": <radius>,
                "height": <height>,
                "position": <position>,
                "orientation": <orientation>
            }
        ],
        ...
    }

Generated destination directory format:
    <args.dst_dir>/
        <sid>.p

Generated destination <sid>.p file format:
    {
        "objects": {
            <oid>: {
                "shape": <shape>,
                "color": <color>
                "radius": <radius>,
                "height": <height>,
                "position": <position>,
                "orientation": <orientation>
            }
        },
        ...
    }
"""
import argparse
import os
import pprint
from tqdm import tqdm

import scene.generator
import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    util.delete_and_create_dir(dir=args.dst_dir)

    # Load scene parameters for sampling attributes.
    scene_params = util.load_json(args.scene_json_path)

    # Loop over each state in the directory.
    n_trials, n_states, n_objects = 0, 0, 0
    for t in tqdm(sorted(os.listdir(args.src_dir))):
        src_t_dir = os.path.join(args.src_dir, t)
        dst_t_dir = os.path.join(args.dst_dir, t)
        util.delete_and_create_dir(dir=dst_t_dir)
        n_trials += 1
        for fname in sorted(os.listdir(src_t_dir)):
            # Load the state.
            src_path = os.path.join(src_t_dir, fname)
            state = util.load_pickle(path=src_path)

            # Loop over the objects.
            new_state_objects = []
            for odict in state["objects"]:
                # Assign random color.
                odict["color"] = scene.generator.gen_rand_color(scene_params["colors"])
                new_state_objects.append(odict)
                n_objects += 1

            # Override the object states with updated states.
            state["objects"] = new_state_objects

            # Save the state.
            dst_path = os.path.join(dst_t_dir, fname)
            util.save_pickle(path=dst_path, data=state)
            n_states += 1
    print(
        f"Number of updated entities:\n"
        f"\tTrials: {n_trials}\n"
        f"\tStates: {n_states}\n"
        f"\tObjects: {n_objects}\n"
    )


if __name__ == "__main__":
    print("*****complete_states.py*****")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        required=True,
        type=str,
        help="The source directory containing partial states from placing.",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        type=str,
        help="The destination directory to save the full, complete states.",
    )
    parser.add_argument(
        "--scene_json_path",
        required=True,
        type=str,
        help="The path to the JSON file that contains scene parameters to sample from.",
    )
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args=args)
