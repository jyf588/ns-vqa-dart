import json
import os
import pickle
import sys
import time
from typing import *

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DashTorchDataset(Dataset):
    def __init__(self, paths: List[str], height: int, width: int):
        """A Pytorch Dataset for DASH objects.

        Args:
            paths: A list of pickle filepaths containing example data.
            height: The height to resize the image to.
            width: The width to resize the image to.
        
        Attributes:
            height: The height to resize the image to.
            width: The width to resize the image to.
            transforms: The transform to apply to the loaded images.
        """
        self.paths = paths
        self.height = height
        self.width = width

        self.normalize = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
        ]

        print(f"Initialized DashTorchDataset containing {len(self)} examples.")

    def __len__(self) -> int:
        """Gets the total number of examples in the dataset.
        
        Returns:
            n_examples: The number of examples in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, idx: int):
        """Loads a single example from the dataset.

        Args:
            idx: The example index to load.
        
        Returns:
            X: The input data, which contains a cropped image of the object
                concatenated with the original image of the scene, with the
                object cropped out.
            y: Labels for the example.
        """
        path = self.paths[idx]
        with open(path, "rb") as f:
            X, y, sid, oid, path = pickle.load(f)

        X = transforms.Compose(self.normalize)(X)
        return X, y, sid, oid, path
