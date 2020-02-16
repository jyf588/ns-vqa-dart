import cv2
import imageio
import json
import numpy as np
import os
from typing import *

from bullet.camera import BulletCamera
import bullet.dash_object
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

    """ Scene examples. """

    def load_example(self, eid: str) -> Tuple[Any]:
        """Loads data for a specified example.

        Args:
            eid: The example ID.
        
        Returns:
            objects: A list of DashObjects in the scene.
            camera: The camera of the scene.
            rgb: The RGB image of the scene.
            mask: The mask of the scene.
        """
        objects, camera = self.load_labels(eid=eid)
        rgb = self.load_rgb(eid=eid)
        mask = self.load_mask(eid=eid)
        return objects, camera, rgb, mask

    def save_example(
        self,
        objects: List[DashObject],
        camera: BulletCamera,
        rgb: np.ndarray,
        mask: np.ndarray,
    ):
        """Saves an example scene.

        Args:
            objects: A list of DashObjects in the scene.
            camera: The camera of the scene.
            rgb: The RGB image of the scene.
            mask: The mask of the scene.
        """
        self.save_rgb(rgb=rgb, eid=self.eid)
        self.save_mask(mask=mask, eid=self.eid)
        self.save_labels(objects=objects, camera=camera, eid=self.eid)
        self.eids.append(self.eid)
        self.save_example_ids(eids=self.eids)
        self.eid += 1

    """ Objects. """

    def load_object(self, img_id: int, oid: int) -> DashObject:
        objects, camera = self.load_labels(eid=img_id)
        debug = 0
        for o in objects:
            if o.img_id == img_id and o.oid == oid:
                return o
        raise ValueError(
            f"Could not local object with image ID {img_id} and object ID {oid}."
        )

    def load_objects(
        self,
        min_img_id: Optional[int] = None,
        max_img_id: Optional[int] = None,
    ) -> List[DashObject]:
        """Loads a list of DashObjects that fall within the bounds of image 
        IDs.

        Args:
            min_img_id: The minimum image ID to include.
            max_img_id: The maximum image ID to include.
        
        Returns:
            objects: A list of DashObject's.
        """
        scene_ids = self.load_example_ids()
        all_objects = []
        for sid in scene_ids:
            objects, camera = self.load_labels(eid=sid)
            all_objects += objects
        return objects

    def load_object_xy(
        self,
        o: DashObject,
        use_attr: bool,
        use_position: bool,
        use_up_vector: bool,
        use_height: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads xy data for a given object.

        Args:
            o: The DashObject.
            use_attr: Whether to use attributes in the label.
            use_position: Whether to use position in the label.
            use_up_vector: Whether to use the up vector in the label.
            use_height: Whether to use height in the label.
        
        Returns:
            data: The input data, which contains a cropped image of the object
                concatenated with the original image of the scene, with the
                object cropped out. (RGB, HWC)
            y: Labels for the example.
        """
        rgb = self.load_rgb(eid=o.img_id)
        mask = self.load_mask(eid=o.img_id)
        data = self.compute_data_from_rgb_and_mask(o=o, rgb=rgb, mask=mask)

        y = o.to_y_vec(
            use_attr=use_attr,
            use_position=use_position,
            use_up_vector=use_up_vector,
            use_height=use_height,
        )
        return data, y

    def compute_data_from_rgb_and_mask(
        self, o: DashObject, rgb: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Constructs the data tensor for an object.

        Args:
            o: The DashObject.
            rgb: The RGB image of the entire scene.
            mask: A 2D mask where each pixel holds the object ID it belongs to.
        
        Returns:
            data: The final data, which contains a cropped image of the object
                concatenated with the original image of the scene, with the
                object cropped out. (RGB, HWC)
        """
        bbox = o.compute_bbox(mask)
        data = np.zeros((480, 480, 6)).astype(np.uint8)
        if bbox is None:
            pass
            print("Bbox is None")
        else:
            x, y, w, h = bbox
            seg = rgb[y : y + h, x : x + w, :].copy()
            rgb[y : y + h, x : x + w, :] = 0.0
            seg = cv2.resize(seg, (480, 480), interpolation=cv2.INTER_AREA)
            data[:, :, :3] = seg
        data[80:400, :, 3:6] = rgb
        return data

    """ Image IDs. """

    def load_example_ids(self) -> List[int]:
        """Gets all of the example ids that exist in the dataset dir.
        
        Returns:
            eids: A list of example IDs.
        """
        eids = bullet.util.load_json(path=self.construct_path(key="eids"))
        return eids

    def save_example_ids(self, eids: List[int]):
        """Saves a list of example IDs that exist in the dataset.
        
        Args:
            eids: A list of example IDs to save.
        """
        bullet.util.save_json(path=self.construct_path(key="eids"), data=eids)

    """ Scene annotations. """

    def load_labels(self, eid: int) -> Dict:
        """Loads the ground truth labels for a single example.

        Args:
            eid: The example ID.

        Returns:
            objects: A list of DashObject's.
            camera: A BulletCamera.
        """
        json_dict = bullet.util.load_json(
            path=self.construct_path(key="json", eid=eid)
        )
        objects = []
        for odict in json_dict["objects"]:
            objects.append(bullet.dash_object.from_json(odict))
        camera = bullet.camera.from_json(json_dict["camera"])
        return objects, camera

    def save_labels(
        self, objects: List[DashObject], camera: BulletCamera, eid: str
    ):
        """Saves a json dictionary of camera and object information.
        
        Args:
            objects: A list of DashObject's.
            camera: A BulletCamera.
            eid: The example ID.
        """
        json_dict = {"camera": camera.to_json(), "objects": []}

        for o in objects:
            # Store the image ID that the object corresponds to.
            o.img_id = eid
            json_dict["objects"].append(o.to_json())

        bullet.util.save_json(
            path=self.construct_path(key="json", eid=eid), data=json_dict
        )

    """ RGB functions. """

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

    """ Mask functions. """

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

    """ Path construction. """

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

