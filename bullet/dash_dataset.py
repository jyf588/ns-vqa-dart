import imageio
import json
import numpy as np
import os
from typing import *

from bullet.camera import BulletCamera
from bullet.dash_object import DashObject


class DashDataset:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.image_id = 0

    def save_example(
        self,
        objects: List[DashObject],
        camera: BulletCamera,
        rgb: np.ndarray,
        mask: np.ndarray,
    ):
        self.save_labels(objects=objects, camera=camera)
        self.save_rgb(rgb)
        self.save_mask(mask)
        self.image_id += 1

    def save_labels(self, objects: List[DashObject], camera: BulletCamera):
        json_dict = {}

        json_dict = {"camera": camera.to_json_dict(), "objects": []}

        for o in objects:
            json_dict["objects"].append(o.to_json_dict())

        # with open(path, "w") as f:
        #     json.dump(
        #         json_dict, f, sort_keys=True, indent=2, separators=(",", ": ")
        #     )

    def save_rgb(self, rgb: np.ndarray):
        path = os.path.join(self.dataset_dir, "rgb", f"{self.image_id:05}.png")
        # imageio.imwrite(path, rgb)

    def save_mask(self, mask: np.ndarray):
        # print(
        #     f"Mask array type: {mask.dtype}\tmin: {mask.min()}\tmax: {mask.max()}"
        # )
        path = os.path.join(
            self.dataset_dir, "mask", f"{self.image_id:05}.npy"
        )
        # np.save(path, mask)
