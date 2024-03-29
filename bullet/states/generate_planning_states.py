"""Generates states for planning. Note that planning states only include 
objects and no robot states.

Structure of generated output files:
    <args.output_dir>/
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
            }
        }
"""
import argparse
import os
import pprint
from tqdm import tqdm
from typing import *

import scene.options as scene_options
import scene.generate as scene_gen
import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    util.delete_and_create_dir(args.dst_dir)

    # Create options based on the JSON configurations.
    task2opt, task2gen_opt = scene_options.create_options(args.scene_json_path)

    tasks = ["stack", "place"]
    n_tasks = len(tasks)
    assert args.n_examples % n_tasks == 0
    n_examples_per_task = args.n_examples // n_tasks

    # Create the scenes for each task, using the options for each task.
    task2scenes = {}
    for task in ["stack", "place"]:
        opt = task2opt[task]
        # Create the scene generator. Seeds are specified at the task-level.
        type2gen = scene_gen.create_generators(
            seed=opt["seed"], type2gen_opt=task2gen_opt[task]
        )
        task_scenes = scene_gen.generate_scenes(
            n_scenes=n_examples_per_task, type2gen=type2gen
        )
        task2scenes[task] = task_scenes

    for task, scenes in task2scenes.items():
        for idx, scene in enumerate(scenes):
            state = {"objects": scene}
            eid = f"{task}_{idx:06}"
            util.save_pickle(path=os.path.join(args.dst_dir, f"{eid}.p"), data=state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_json_path",
        required=True,
        type=str,
        help="The path to the JSON file specifying scene sampling parameters.",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        type=str,
        help="The directory to generate files to.",
    )
    parser.add_argument(
        "--n_examples",
        required=True,
        type=int,
        help="The number of examples to generate.",
    )
    args = parser.parse_args()
    main(args=args)
