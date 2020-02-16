import imageio
import json
import numpy as np
import os
from typing import *

from bullet.camera import BulletCamera
from bullet.dash_object import DashObject

KEY2EXT = {"rgb": "png", "mask": "npy", "json": "json"}


class DashDataset:
    def __init__(self, dataset_dir: str):
        """
        Args:
            dataset_dir: The directory to save the dataset in.
        
        Attributes:
            dataset_dir: The directory to save the dataset in.
            eid: The example ID to assign to the next call to 
                `self.save_example`.
        """
        self.dataset_dir = dataset_dir
        self.eid = 0
        self.eids = []

    def load_example(self, eid: str) -> Tuple[Any]:
        """Loads data for a specified example.

        Args:
            eid: The example ID.
        
        Returns:
            json_dict: The JSON dictionary containing the labels.
            rgb: The RGB image for the example.
            mask: The mask for the example.
        """
        json_dict = self.load_labels(eid=eid)
        rgb = self.load_rgb(eid=eid)
        mask = self.load_mask(eid=eid)
        return json_dict, rgb, mask

    def save_example(
        self,
        objects: List[DashObject],
        camera: BulletCamera,
        rgb: np.ndarray,
        mask: np.ndarray,
    ):
        self.save_rgb(rgb=rgb, eid=self.eid)
        self.save_mask(mask=mask, eid=self.eid)
        self.save_labels(objects=objects, camera=camera, eid=self.eid)
        self.eids.append(self.eid)
        self.save_example_ids(eids=self.eids)
        self.eid += 1

    def load_example_ids(self) -> List[int]:
        """Gets all of the example ids that exist in the dataset dir.
        
        Returns:
            eids: A list of example IDs.
        """
        eids = self.load_json(path=self.construct_path(key="eids"))
        return eids

    def save_example_ids(self, eids: List[int]):
        """Saves a list of example IDs that exist in the dataset.
        
        Args:
            eids: A list of example IDs to save.
        """
        self.save_json(path=self.construct_path(key="eids"), data=eids)

    def load_labels(self, eid: int) -> Dict:
        """Loads the ground truth labels for a single example.

        Args:
            eid: The example ID.

        Returns:
            json_dict: The JSON dictionary containing the labels.
        """
        json_dict = self.load_json(
            path=self.construct_path(key="json", eid=eid)
        )
        return json_dict

    def save_labels(
        self, objects: List[DashObject], camera: BulletCamera, eid: str
    ):
        """Saves a json dictionary of camera and object information.
        
        Args:
            objects: A list of DashObject's.
            camera: A BulletCamera.
            eid: The example ID.
        """
        json_dict = {"camera": camera.to_json_dict(), "objects": []}

        for o in objects:
            json_dict["objects"].append(o.to_json_dict())

        self.save_json(
            path=self.construct_path(key="json", eid=eid), data=json_dict
        )

    def load_rgb(self, eid: int) -> np.ndarray:
        """Loads a RGB image for an example ID.
        
        Args:
            eid: The example ID.
        
        Returns:
            rgb: The RGB image for the example.
        """
        rgb = imageio.imread(self.construct_path(key="rgb", eid=eid))
        return rgb

    def save_rgb(self, rgb: np.ndarray, eid: int):
        """Saves a RGB image under an example ID.
        
        Args:
            rgb: The RGB image to save.
            eid: The example ID.
        """
        imageio.imwrite(self.construct_path(key="rgb", eid=eid), rgb)

    def load_mask(self, eid: str):
        """Loads a mask for an example ID.
        
        Args:
            eid: The example ID.
        
        Returns:
            mask: The mask for the example.
        """
        return np.load(self.construct_path(key="mask", eid=eid))

    def save_mask(self, mask: np.ndarray, eid: str):
        """Saves a mask under an example ID.
        
        Args:
            mask: The mask to save.
            eid: The example ID.
        """
        np.save(self.construct_path(key="mask", eid=eid), mask)

    def construct_path(self, key: str, eid: Optional[int] = None) -> str:
        """Constructs the filepath for a given key and example.

        Args:
            key: The key of the file type, e.g., "rgb", "mask", "json".
            eid: The example ID.

        Returns:
            path: The path to the requested file.
        """
        if key == "eids":
            path = os.path.join(self.dataset_dir, f"eids.json")
        else:
            key_dir = os.path.join(self.dataset_dir, key)
            os.makedirs(key_dir, exist_ok=True)
            path = os.path.join(key_dir, f"{eid:05}.{KEY2EXT[key]}")
        return path

    def load_json(self, path: str) -> Any:
        """Loads a JSON file.

        Args:
            path: The path to the JSON file.
        
        Returns:
            data: The JSON data.
        """
        with open(path, "r") as f:
            data = json.load(f)
            return data

    def save_json(self, path: str, data: Any):
        with open(path, "w") as f:
            json.dump(
                data, f, sort_keys=True, indent=2, separators=(",", ": ")
            )

