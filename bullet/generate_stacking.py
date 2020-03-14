"""Generates a stacking dataset from saved states from running enjoy.py."""
import os
import random
from tqdm import tqdm
from typing import *

from camera import BulletCamera
from dash_dataset import DashDataset
from dash_object import DashObject, DashTable
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
        self.n_examples_per_state = 10

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
            for fname in os.listdir(state_dir):
                path = os.path.join(state_dir, fname)
                paths.append(path)

            # Sample a subset.
            random.shuffle(paths)
            paths = paths[: self.n_examples_per_state]

            dataset_paths += paths

        for path in tqdm(dataset_paths):
            self.generate_example(path=path)

    def generate_example(self, path: str):
        odicts = util.load_pickle(path=path)

        state_objects = []
        for odict in odicts:
            odict["oid"], odict["img_id"] = None, None
            odict["color"] = bullet_renderer.gen_rand_obj_color()
            o = dash_object.from_json(odict)
            state_objects.append(o)

        surround_objects = self.gen_surround_objects(
            existing_objects=state_objects
        )

        objects = state_objects + surround_objects

        # Render objects.
        self.renderer.render_objects(
            objects=objects, position_mode=self.position_mode
        )

        self.dataset.save_example(objects=objects, camera=self.camera)
        self.renderer.remove_objects(objects=objects)

    def gen_surround_objects(self, existing_objects: List[DashObject]):
        """Generates random surrounding objects.

        Returns:
            objects: A list of randomly object dictionaries of surrounding
                objects.
        """
        objects = self.objects_generator.generate_tabletop_objects(
            existing_objects=existing_objects
        )
        return objects


if __name__ == "__main__":
    generator = StackingDatasetGenerator()
    generator.run()
