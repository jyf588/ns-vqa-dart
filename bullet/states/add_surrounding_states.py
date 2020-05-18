"""Updates states by adding surrounding objects to existing objects.

Expected structure of <args.src_dir>:
    <args.src_dir>/
        <id>.p = {
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

Generated output structure of <args.dst_dir>:
    <args.dst_dir>/
        <id>.p = {
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
from typing import *

import ns_vqa_dart.bullet.dash_object as dash_object
from scene.generator import SceneGenerator
import scene.options
import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    util.delete_and_create_dir(dir=args.dst_dir)

    # Create options based on the JSON configurations.
    task2opt, task2gen_opt = scene.options.create_options(args.scene_json_path)
    opt = task2opt[args.task]
    surround_gen_opt_list = task2gen_opt[args.task]["surround"]
    assert len(surround_gen_opt_list) == 1
    surround_gen_opt = surround_gen_opt_list[0]

    # Create the scene generator.
    scene_generator = SceneGenerator(seed=opt["seed"], opt=surround_gen_opt)

    scene_count = 0
    manip_count = 0
    surround_count = 0

    # Loop over states in the source directory.
    for t in tqdm(sorted(os.listdir(args.src_dir))):
        src_t_dir = os.path.join(args.src_dir, t)
        dst_t_dir = os.path.join(args.dst_dir, t)
        util.delete_and_create_dir(dst_t_dir)
        for f in sorted(os.listdir(src_t_dir)):
            scene_count += 1

            # Load the current state file.
            state = util.load_pickle(path=os.path.join(src_t_dir, f))

            # Extract the existing objects in the state, which are manipulable objects.
            manip_odicts = state["objects"]
            manip_count += len(manip_odicts)

            # Generate random surrounding objects around existing objects.
            surround_odicts = scene_generator.generate_tabletop_objects(
                existing_odicts=manip_odicts, unique_odicts=manip_odicts
            )
            surround_count += len(surround_odicts)

            # Add to existing list of objects.
            state["objects"] += surround_odicts

            # Save the new state file into the destination directory.
            util.save_pickle(path=os.path.join(dst_t_dir, f), data=state)

    print(f"Number of scenes: {scene_count}")
    print(f"Manip count: {manip_count}")
    print(f"Surround count: {surround_count}")


if __name__ == "__main__":
    print("*****add_surrounding_states.py*****")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, type=str, help="The task to generate states for.",
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        type=str,
        help="The directory containing original states.",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        type=str,
        help="The directory containing original states.",
    )
    parser.add_argument(
        "--scene_json_path",
        required=True,
        type=str,
        help="The path to the JSON file specifying scene sampling parameters.",
    )
    args = parser.parse_args()

    pprint.pprint(vars(args))
    main(args)
