import json
import os
import sys
from typing import *

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

sys.path.append("/home/michelle/workspace/ns-vqa-dart")
from bullet.dash_dataset import DashDataset


class DashTorchDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        use_attr: bool,
        use_position: bool,
        use_up_vector: bool,
        use_height: bool,
        min_img_id: Optional[int] = None,
        max_img_id: Optional[int] = None,
    ):
        """A Pytorch Dataset for DASH objects.

        Args:
            dataset_dir: The directory to load data from.
            use_attr: Whether to use attributes in the label.
            use_position: Whether to use position in the label.
            use_up_vector: Whether to use the up vector in the label.
            use_height: Whether to use height in the label.
            min_img_id: The minimum image ID to include in the dataset.
            max_img_id: The maximum image ID to include in the dataset.
        
        Attributes:
            objects: A list of DashObjects.
            use_attr: Whether to use attributes in the label.
            use_position: Whether to use position in the label.
            use_up_vector: Whether to use the up vector in the label.
            use_height: Whether to use height in the label.
            transforms: The transform to apply to the loaded images.
        """
        self.dataset = DashDataset(dataset_dir=dataset_dir)

        # Load object examples included in the image ID bounds.
        self.objects = self.dataset.load_objects(
            min_img_id=min_img_id, max_img_id=max_img_id
        )

        self.use_attr = use_attr
        self.use_position = use_position
        self.use_up_vector = use_up_vector
        self.use_height = use_height

        self.normalize = [
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225]
            )
        ]
        self.transform = [transforms.ToTensor()]

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
        data, y = self.dataset.load_object_xy(
            o=self.objects[idx],
            use_attr=self.use_attr,
            use_position=self.use_position,
            use_up_vector=self.use_up_vector,
            use_height=self.use_height,
        )
        data = transforms.Compose(self.transform)(data)
        data[:3] = transforms.Compose(self.normalize)(data[:3])
        data[3:6] = transforms.Compose(self.normalize)(data[3:6])
        y = torch.Tensor(y)
        return data, y

