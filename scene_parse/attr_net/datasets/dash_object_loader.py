import json
import os
import sys
import time
from typing import *

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

sys.path.append("/home/michelle/workspace/ns-vqa-dart")
from bullet.dash_dataset import DashDataset
from bullet.profiler import Profiler


class DashTorchDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        height: int,
        width: int,
        use_attr: bool,
        use_size: bool,
        use_position: bool,
        use_up_vector: bool,
        coordinate_frame: str,
        split: str,
        min_img_id: Optional[int] = None,
        max_img_id: Optional[int] = None,
    ):
        """A Pytorch Dataset for DASH objects.

        Args:
            dataset_dir: The directory to load data from.
            height: The height to resize the image to.
            width: The width to resize the image to.
            use_attr: Whether to include attributes in the label.
            use_size: Whether to include size (radius and height) in the label.
            use_position: Whether to include position in the label.
            use_up_vector: Whether to include the up vector in the label.
            coordinate_frame: The coordinate frame to train on, either "world"
                or "camera" coordinate frame.
            min_img_id: The minimum image ID to include in the dataset.
            max_img_id: The maximum image ID to include in the dataset.
            split: The name of the split (i.e., train, val, or test)
        
        Attributes:
            dataset: The DashDataset for loading objects.
            height: The height to resize the image to.
            width: The width to resize the image to.
            objects: A list of DashObjects.
            use_attr: Whether to include attributes in the label.
            use_size: Whether to include size (radius and height) in the label.
            use_position: Whether to include position in the label.
            use_up_vector: Whether to include the up vector in the label.
            transforms: The transform to apply to the loaded images.
            exclude_y: Whether to exclude loading y labels. Currently this is
                true if the split is "test".
        """
        print(f"Initializing DashTorchDataset...")
        self.dataset = DashDataset(dataset_dir=dataset_dir)

        self.height = height
        self.width = width

        print(f"min_img_id: {min_img_id}")
        print(f"max_img_id: {max_img_id}")

        # Load object examples included in the image ID bounds.
        self.objects = self.dataset.load_objects(
            exclude_out_of_view=False,
            min_img_id=min_img_id,
            max_img_id=max_img_id,
        )

        self.use_attr = use_attr
        self.use_size = use_size
        self.use_position = use_position
        self.use_up_vector = use_up_vector
        self.coordinate_frame = coordinate_frame

        self.transform_to_tensor = [transforms.ToTensor()]
        # self.transforms = [
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
        # ]

        self.transforms = [
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.225] * 3),
        ]

        self.exclude_y = split == "test"

        print(f"Initialized DashTorchDataset containing {len(self)} examples.")

        self.profiler = Profiler()

    def __len__(self) -> int:
        """Gets the total number of examples in the dataset.
        
        Returns:
            n_examples: The number of examples in the dataset.
        """
        n_examples = len(self.objects)
        return n_examples

    def __getitem__(self, idx: int):
        """Loads a single example from the dataset.

        Args:
            idx: The example index to load.
        
        Returns:
            data: The input data, which contains a cropped image of the object
                concatenated with the original image of the scene, with the
                object cropped out.
            y: Labels for the example.
        """
        o = self.objects[idx]

        data = self.dataset.load_object_x(o=o)
        data = transforms.Compose(self.transform_to_tensor)(data)
        x = torch.zeros(6, self.height, self.width)
        x[:3] = transforms.Compose(self.transforms)(data[:3])
        x[3:6] = transforms.Compose(self.transforms)(data[3:6])

        if self.exclude_y:
            return data, o.img_id, o.oid
        else:
            y = self.dataset.load_object_y(
                o=o,
                use_attr=self.use_attr,
                use_size=self.use_size,
                use_position=self.use_position,
                use_up_vector=self.use_up_vector,
                coordinate_frame=self.coordinate_frame,
            )
            y = torch.Tensor(y)
            return data, y, o.img_id, o.oid
