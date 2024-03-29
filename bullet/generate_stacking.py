"""Generates a stacking dataset from saved states from running enjoy.py."""
import os
import random
from tqdm import tqdm
from typing import *

from camera import BulletCamera
from dash_dataset import DashDataset
from dash_object import DashObject, DashTable, DashRobot
import dash_object, util
from random_objects import RandomObjectsGenerator
from renderer import BulletRenderer
import renderer as bullet_renderer


class StackingDatasetGenerator:
    def __init__(self):
        seed = 0
        random.seed(seed)

        # Paths
        dataset_dir = "/home/michelle/datasets/stacking_v003"
        self.state_dirs = [
            "/home/michelle/datasets/delay_box_states",
            "/home/michelle/datasets/delay_cyl_states",
        ]
        self.n_examples_per_state = 11000

        bc = util.create_bullet_client(mode="direct")

        self.renderer = BulletRenderer(p=bc)
        self.table_object = DashTable(position=[0.2, 0.2, 0.0])
        table_y_offset = self.table_object.position[1]
        self.renderer.render_object(o=self.table_object, position_mode="base")
        self.position_mode = "com"
        self.objects_generator = RandomObjectsGenerator(
            seed=seed,
            n_objs_bounds=(0, 5),
            obj_dist_thresh=0.14,
            max_retries=50,
            shapes=["box", "cylinder", "sphere"],
            colors=["red", "blue", "yellow", "green"],
            radius_bounds=(0.01, 0.07),
            height_bounds=(0.05, 0.20),
            x_bounds=(0.0, 0.4),
            y_bounds=(-0.5 + table_y_offset, 0.5 + table_y_offset),
            z_bounds=(0.0, 0.0),
            position_mode=self.position_mode,
        )

        # Real arm
        self.robot = DashRobot(
            p=bc,
            urdf_path="my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_revolute_head.urdf",
            position=[-0.3, 0.5, -1.25],
        )

        # Green arm
        # self.robot = DashRobot(
        #     p=bc,
        #     urdf_path="my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_2.urdf",
        #     position=[-0.30, 0.348, 0.272],
        #     include_head=False,
        # )

        # Camera
        self.camera = BulletCamera(
            p=bc,
            position=[-0.2237938867122504, 0.0, 0.5425],
            rotation=[0.0, 50.0, 0.0],
            offset=[0.0, table_y_offset, 0.0],
        )

        self.dataset = DashDataset(dataset_dir=dataset_dir)

    def run(self):
        dataset_paths = []
        for state_dir in self.state_dirs:

            paths = []
            for fname in sorted(os.listdir(state_dir)):
                path = os.path.join(state_dir, fname)
                paths.append(path)

            # Sample a subset.
            random.shuffle(paths)
            paths = paths[: self.n_examples_per_state]

            dataset_paths += paths

        # Shuffle the paths so that box and cyl are interleaved.
        random.shuffle(dataset_paths)

        for path in tqdm(dataset_paths):
            self.generate_example(path=path)

    def generate_example(self, path: str):
        state = util.load_pickle(path=path)

        # Randomly set object colors if it's not already set.
        for i in range(len(state["objects"])):
            odict = state["objects"][i]
            if "color" not in odict:
                odict["color"] = gen_rand_obj_color()
            state["objects"][i] = odict

        # Generate surrounding objects that don't clash with existing objects.
        surround_odicts = self.gen_surround_objects(
            existing_odicts=state["objects"]
        )

        # Add surrounding objects to the list of state objects.
        state["objects"] = state["objects"] + surround_odicts

        self.renderer.reconstruct_from_state(state=state)

        self.dataset.save_example(objects=objects, camera=self.camera)
        self.renderer.remove_objects(objects=objects)

    def gen_surround_objects(self, existing_odicts: List[Dict]) -> List[Dict]:
        """Generates random surrounding objects.

        Args:
            existing_odicts: A list of dictionaries of objects that already
                exist.

        Returns:
            odicts: A list of randomly generated object dictionaries of
                surrounding objects.
        """
        odicts = self.objects_generator.generate_tabletop_objects(
            existing_odicts=existing_odicts
        )
        return odicts


if __name__ == "__main__":
    generator = StackingDatasetGenerator()
    generator.run()
