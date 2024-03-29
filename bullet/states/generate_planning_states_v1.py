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

import ns_vqa_dart.bullet.dash_object as dash_object
from ns_vqa_dart.bullet.random_objects import RandomObjectsGenerator
import ns_vqa_dart.bullet.util as util
import my_pybullet_envs.utils as env_utils


def main(args: argparse.Namespace):
    util.delete_and_create_dir(args.dst_dir, exist_ok=True)

    # Create options based on the JSON configurations.
    task2opt, task2gen_opt = scene.options.create_options(args.scene_json_path)
    opt = task2opt[args.task]
    surround_gen_opt_list = task2gen_opt[args.task]["surround"]
    assert len(surround_gen_opt_list) == 1
    surround_gen_opt = surround_gen_opt_list[0]

    objects_generator = RandomObjectsGenerator(
        seed=1,
        n_objs_bounds=(2, 7),
        obj_dist_thresh=0.1,
        max_retries=50,
        shapes=["box", "cylinder", "sphere"],
        radius_bounds=(0.01, 0.07),
        height_bounds=(0.05, 0.20),
        x_bounds=(env_utils.TX_MIN, env_utils.TX_MAX),
        y_bounds=(env_utils.TY_MIN, env_utils.TY_MAX),
        z_bounds=(0.0, 0.0),
        position_mode="com",
    )

    for sid in tqdm(range(args.n_examples)):
        odicts = objects_generator.generate_tabletop_objects()
        oid2odicts = dash_object.assign_ids_to_odicts(odicts=odicts, start_id=0)
        state = {"objects": oid2odicts}
        util.save_pickle(path=os.path.join(args.dst_dir, f"{sid:06}.p"), data=state)


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
