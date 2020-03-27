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
from tqdm import tqdm
from typing import *

from ns_vqa_dart.bullet.random_objects import RandomObjectsGenerator
import ns_vqa_dart.bullet.util as util
import my_pybullet_envs.utils as env_utils


def main(args: argparse.Namespace):
    objects_generator = RandomObjectsGenerator(
        seed=1,
        n_objs_bounds=(0, 6),
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
    # Loop over states in the source directory.
    for fname in tqdm(os.listdir(args.src_dir)):
        # Load the current state file.
        src_state = util.load_pickle(path=os.path.join(args.src_dir, fname))
        src_oid2odict = src_state["objects"]
        src_oids = list(src_oid2odict.keys())
        src_odicts = list(src_oid2odict.values())

        # Generate random surrounding objects around existing objects.
        new_odicts = objects_generator.generate_tabletop_objects(
            existing_odicts=src_odicts
        )

        # Assign object IDs to the new objects.
        # Object IDs are assigned starting after the maximum existing
        # object ID in the source objects.
        oid2odicts = assign_ids_to_odicts(
            odicts=new_odicts, start_id=max(src_oids) + 1
        )

        # Combine new objects with existing objects.
        for oid, odict in oid2odicts.items():
            src_state["objects"][oid] = odict

        # Save the new state file into the destination directory.
        util.save_pickle(
            path=os.path.join(args.dst_dir, fname), data=src_state
        )


def assign_ids_to_odicts(odicts: List[Dict], start_id: int):
    """Assigns object IDs to object dictionaries.

    Args:
        odicts: A list of object dictionaries, with the format:
            [
                {
                    <attr>: <value>
                }
            ]
        start_id: The starting ID to assign object IDs.
    
    Returns:
        oid2odict: A mapping from assigned object ID to dictionary, with the 
            format: {
                <oid>: {
                    <attr>: <value>
                }
            }
    """
    next_id_to_assn = start_id
    oid2odict = {}
    for odict in odicts:
        oid = next_id_to_assn
        oid2odict[oid] = odict
        next_id_to_assn += 1
    return oid2odict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    main(args)
