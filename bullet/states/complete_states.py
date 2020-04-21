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

import ns_vqa_dart.bullet.random_objects as random_objects
import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    util.delete_and_create_dir(dir=args.dst_dir)

    # Loop over each state in the directory.
    for state_fname in tqdm(os.listdir(args.src_dir)):
        # Load the state.
        src_path = os.path.join(args.src_dir, state_fname)
        state = util.load_pickle(path=src_path)

        # Loop over the objects.
        new_state_objects = {}
        for idx, odict in enumerate(state["objects"]):
            # Assign oid. We arbitrarily use the idx of the object as the
            # object ID. We want the object IDs to be between [0, max_objects)
            # because when renderering the objects later on in Unity, Unity
            # requires that object tags are pre-defined as a fixed set of
            # possible object IDs.
            oid = idx

            # Assign random color.
            odict["color"] = random_objects.generate_random_color()

            new_state_objects[oid] = odict

        # Override the object states.
        state["objects"] = new_state_objects

        # Save the state.
        dst_path = os.path.join(args.dst_dir, state_fname)
        util.save_pickle(path=dst_path, data=state)


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
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args=args)
