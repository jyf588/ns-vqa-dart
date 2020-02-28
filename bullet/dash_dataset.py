import cv2
import imageio
import json
import numpy as np
import os
import time
from tqdm import tqdm
from typing import *

from bullet.camera import BulletCamera
import bullet.dash_object
from bullet.dash_object import DashObject
from bullet.profiler import Profiler

KEY2EXT = {"rgb": "png", "mask": "npy", "json": "json", "input_data": "npy"}


class DashDataset:
    def __init__(self, dataset_dir: str, threshold_obj_area=0):
        """
        Args:
            dataset_dir: The directory to save the dataset in.
            threshold_obj_area: The threshold object area to include in the 
                dataset.
        
        Attributes:
            dataset_dir: The directory to save the dataset in.
            threshold_obj_area: The threshold object area to include in the 
                dataset.
            eid: The example ID to assign to the next call to 
                `self.save_example`.
        """
        self.dataset_dir = dataset_dir
        self.threshold_obj_area = threshold_obj_area
        self.eid = 0
        self.eids = []
        self.profiler = Profiler()

    """ Scene examples. """

    def load_example_for_eid(self, eid: str) -> Tuple[Any]:
        """Loads data for a specified example.

        Args:
            eid: The example ID.
        
        Returns:
            objects: A list of DashObjects in the scene.
            camera: The camera of the scene.
            rgb: The RGB image of the scene.
            mask: The mask of the scene.
        """
        objects, camera = self.load_objects_and_camera_for_eid(eid=eid)
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
        # Filter out objects that are out-of-view.
        objects = self.filter_objects_by_area(
            objects=objects, mask=mask, threshold_area=self.threshold_obj_area
        )

        # Associate each object with the image ID.
        for o in objects:
            o.img_id = self.eid

        self.save_rgb(rgb=rgb, eid=self.eid)
        self.save_mask(mask=mask, eid=self.eid)

        self.save_input_data(objects=objects, rgb=rgb, mask=mask)
        self.save_labels(objects=objects, camera=camera, eid=self.eid)
        self.eids.append(self.eid)
        self.save_example_ids(eids=self.eids)
        self.eid += 1

    """ Objects. """

    def load_object_for_img_id_and_oid(
        self, img_id: int, oid: int
    ) -> DashObject:
        objects, camera = self.load_objects_and_camera_for_eid(eid=img_id)
        debug = 0
        for o in objects:
            if o.img_id == img_id and o.oid == oid:
                return o
        raise ValueError(
            f"Could not locate object with image ID {img_id} and object ID {oid}."
        )

    def load_objects(
        self,
        min_img_id: Optional[int] = None,
        max_img_id: Optional[int] = None,
    ) -> List[DashObject]:
        """Loads a list of DashObjects that fall within the bounds of image 
        IDs.

        Args:
            min_img_id: The minimum image ID, inclusive.
            max_img_id: The maximum image ID, inclusive.
        
        Returns:
            objects: A list of DashObject's.
        """
        print(
            f"Loading objects between example IDs {min_img_id} and {max_img_id} (inclusive)..."
        )
        scene_ids = self.load_example_ids(min_id=min_img_id, max_id=max_img_id)
        all_objects = []
        for sid in tqdm(scene_ids):
            objects = self.load_objects_for_eid(eid=sid)
            all_objects += objects
        return all_objects

    def filter_objects_by_area(
        self, objects: List[DashObject], mask: np.ndarray, threshold_area: int
    ) -> List[DashObject]:
        """Computes the mask for each object and excludes objects with a mask
        area less than a certain threshold.

        Args:
            objects: A list of objects.
            mask: A mask of all the objects in the scene.
            threshold_area: If any objects have an area greater than
                this threshold area, they are included
        
        Returns:
            objects_to_keep: A list of objects to keep.
        """
        objects_to_keep = []
        for o in objects:
            if o.mask_area(mask=mask) > threshold_area:
                objects_to_keep.append(o)
        return objects_to_keep

    def load_object_x(self, o: DashObject) -> np.ndarray:
        """Loads the x data for an object.

        Returns:
            data: The x data.
        """
        data = self.load_input_data(o=o)
        return data

    def load_object_y(
        self,
        o: DashObject,
        use_attr: bool,
        use_size: bool,
        use_position: bool,
        use_up_vector: bool,
        coordinate_frame: str,
    ) -> np.ndarray:
        """Loads y data for a given object.

        Args:
            o: The DashObject.
            use_attr: Whether to use attributes in the label.
            use_position: Whether to use position in the label.
            use_size: Whether to use size in the label.
            use_up_vector: Whether to use the up vector in the label.
            use_height: Whether to use height in the label.
            coordinate_frame: The coordinate frame to use, either "world" or
                "camera" coordinate frame.

        Returns:
            y: Labels for the example.
        """
        camera = self.load_camera_for_eid(eid=o.img_id)
        y = o.to_y_vec(
            use_attr=use_attr,
            use_size=use_size,
            use_position=use_position,
            use_up_vector=use_up_vector,
            coordinate_frame=coordinate_frame,
            camera=camera,
        )
        return y

    def load_input_data(self, o: DashObject) -> np.ndarray:
        """Loads the input data for a given object.

        Args:
            o: A DashObject.
        
        Returns:
            data: The input data for the object.
        """
        data = np.load(self.construct_object_path(o=o, key="input_data"))
        return data

    def save_input_data(
        self, objects: List[DashObject], rgb: np.ndarray, mask: np.ndarray
    ):
        """Save the data that will be used as input to the network.

        Args:
            objects: A list of DashObjects in the scene.
            rgb: The RGB image.
            mask: The mask of the scene.
        """
        for o in objects:
            data = self.compute_data_from_rgb_and_mask(o=o, rgb=rgb, mask=mask)
            np.save(self.construct_object_path(o=o, key="input_data"), data)

    def compute_data_from_rgb_and_mask(
        self,
        o: DashObject,
        rgb: np.ndarray,
        mask: np.ndarray,
        data_height=480,
        data_width=480,
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
        rgb = rgb.copy()
        bbox = o.compute_bbox(mask)
        data = np.zeros((data_height, data_width, 6)).astype(np.uint8)
        input_rgb = rgb.copy()
        if bbox is None:
            print(f"Bbox is None. Object: {o.to_json()}")
        else:
            x, y, w, h = bbox

            # Set the object seg to zeros in the original RGB image.
            input_rgb[mask == o.oid] = 0.0

            # Create the segmentation image (maintain aspect ratio, use
            # replicate padding).
            # First, zero out everything in the RGB image except for the object
            # segmentation.
            rgb_with_only_seg = rgb.copy()
            rgb_with_only_seg[mask != o.oid] = 0.0

            # Crop the segmentation out using its bbox.
            seg = rgb_with_only_seg[y : y + h, x : x + w, :]

            # Compute the new dimensions to resize the segmentation to.
            if h > w:
                aspect_ratio = h / w
                resize_dims = (data_height, int(data_width / aspect_ratio))
            else:
                aspect_ratio = w / h
                resize_dims = (int(data_height / aspect_ratio), data_width)
            H_, W_ = resize_dims

            # Resize the segmentation while maintaining aspect ratio.
            seg = cv2.resize(seg, (W_, H_))  # OpenCV expects WH.
            seg_padded = np.zeros((data_height, data_width, 3), dtype=np.uint8)
            top_pad = (data_height - H_) // 2
            left_pad = (data_width - W_) // 2
            seg_padded[top_pad : top_pad + H_, left_pad : left_pad + W_] = seg

            # Use replicate padding to make the seg image match the desired
            # input data dimensions.
            # top_pad = (data_height - H_) // 2
            # left_pad = (data_width - W_) // 2
            # seg = cv2.copyMakeBorder(
            #     src=seg,
            #     top=top_pad,
            #     bottom=data_height - H_ - top_pad,
            #     left=left_pad,
            #     right=data_width - W_ - left_pad,
            #     borderType=cv2.BORDER_REPLICATE,
            # )
            data[:, :, :3] = seg_padded
        data[80:400, :, 3:6] = input_rgb
        return data

    """ Image IDs. """

    def load_example_ids(
        self, min_id: Optional[int] = None, max_id: Optional[int] = None
    ) -> List[int]:
        """Gets all of the example ids that exist in the dataset dir.
        
        train: (None, 20000) -> 0 through 19999
        test: (20000, None) -> 20000 through 21999

        Args:
            min_id: The minimum image ID, inclusive.
            max_id: The maximum image ID, inclusive.

        Returns:
            filtered_eids: A list of example IDs, optionally filtered to only
                include IDs within the provided ID bounds.
        """
        eids = bullet.util.load_json(
            path=self.construct_scene_path(key="eids")
        )
        if min_id is None:
            min_id = 0
        if max_id is None:
            max_id = len(eids)

        filtered_eids = []
        for eid in eids:
            if min_id <= eid <= max_id:
                filtered_eids.append(eid)
        return filtered_eids

    def save_example_ids(self, eids: List[int]):
        """Saves a list of example IDs that exist in the dataset.
        
        Args:
            eids: A list of example IDs to save.
        """
        bullet.util.save_json(
            path=self.construct_scene_path(key="eids"), data=eids
        )

    """ Scene annotations. """

    def load_objects_and_camera_for_eid(
        self, eid: int
    ) -> Tuple[List[DashObject], BulletCamera]:
        """Loads the ground truth labels for a single example.

        Args:
            eid: The example ID.

        Returns:
            objects: A list of DashObject's.
            camera: A BulletCamera.
        """
        json_dict = bullet.util.load_json(
            path=self.construct_scene_path(key="json", eid=eid)
        )
        objects = []
        for odict in json_dict["objects"]:
            objects.append(bullet.dash_object.from_json(odict))
        camera = bullet.camera.from_json(json_dict["camera"])
        return objects, camera

    def load_objects_for_eid(
        self, eid: int
    ) -> Tuple[List[DashObject], BulletCamera]:
        """Loads the ground truth labels for a single example.

        Args:
            eid: The example ID.

        Returns:
            objects: A list of DashObject's.
        """
        json_dict = bullet.util.load_json(
            path=self.construct_scene_path(key="json", eid=eid)
        )
        objects = []
        for odict in json_dict["objects"]:
            objects.append(bullet.dash_object.from_json(odict))
        return objects

    def load_camera_for_eid(self, eid: int) -> BulletCamera:
        """Loads the camera for a single example.

        Args:
            eid: The example ID.

        Returns:
            camera: The BulletCamera for the example.
        """
        json_dict = bullet.util.load_json(
            path=self.construct_scene_path(key="json", eid=eid)
        )
        camera = bullet.camera.from_json(json_dict["camera"])
        return camera

    def save_labels(
        self, objects: List[DashObject], camera: BulletCamera, eid: int
    ):
        """Saves a json dictionary of camera and object information.
        
        Args:
            objects: A list of DashObject's.
            camera: A BulletCamera.
            eid: The example ID.
        """
        json_dict = {"camera": camera.to_json(), "objects": []}

        for o in objects:
            json_dict["objects"].append(o.to_json())

        bullet.util.save_json(
            path=self.construct_scene_path(key="json", eid=eid), data=json_dict
        )

    """ RGB functions. """

    def load_rgb(self, eid: int) -> np.ndarray:
        """Loads a RGB image for an example ID.
        
        Args:
            eid: The example ID.
        
        Returns:
            rgb: The RGB image for the example.
        """
        rgb = imageio.imread(self.construct_scene_path(key="rgb", eid=eid))
        return rgb

    def save_rgb(self, rgb: np.ndarray, eid: int):
        """Saves a RGB image under an example ID.
        
        Args:
            rgb: The RGB image to save.
            eid: The example ID.
        """
        imageio.imwrite(self.construct_scene_path(key="rgb", eid=eid), rgb)

    """ Mask functions. """

    def load_mask(self, eid: str):
        """Loads a mask for an example ID.
        
        Args:
            eid: The example ID.
        
        Returns:
            mask: The mask for the example.
        """
        return np.load(self.construct_scene_path(key="mask", eid=eid))

    def save_mask(self, mask: np.ndarray, eid: str):
        """Saves a mask under an example ID.
        
        Args:
            mask: The mask to save.
            eid: The example ID.
        """
        np.save(self.construct_scene_path(key="mask", eid=eid), mask)

    """ Path construction. """

    def construct_scene_path(self, key: str, eid: Optional[int] = None) -> str:
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

    def construct_object_path(self, o: DashObject, key: str) -> str:
        """Constructs the path to the input data for the object.

        Args:
            o: The DashObject.
            key: The key of the file type, e.g., "input_data".

        Returns:
            path: The object path.
        """
        path_dir = os.path.join(self.dataset_dir, key, f"{o.img_id:05}")
        os.makedirs(path_dir, exist_ok=True)
        path = os.path.join(path_dir, f"{o.oid:02}.{KEY2EXT[key]}")
        return path
