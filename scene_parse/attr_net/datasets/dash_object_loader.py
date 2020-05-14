import os
import sys
import time
import json
import pickle
import random
import imageio
import numpy as np
from typing import *


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ns_vqa_dart.bullet import util


class DashTorchDataset(Dataset):
    def __init__(self, data_dir: str):
        """A Pytorch Dataset for DASH objects.

        Args:
            data_dir: A folder of data to load from. Should be a folder of pickle files.
        """
        self.data_dir = data_dir

        self.fnames = os.listdir(data_dir)

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
        return len(self.fnames)

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
        X, y = util.load_pickle(os.path.join(self.data_dir, self.fnames[idx]))[:2]

        # X = dash_object.compute_X(img=rgb, mask=mask, keep_occluded=True)
        # y = dash_object.compute_y(
        #     odict=odict,
        #     coordinate_frame=self.opt.coordinate_frame,
        #     cam_position=cam_dict["position"],
        #     cam_orientation=cam_dict["orientation"],
        # )

        X = transforms.Compose(self.normalize)(X)

        return X, y
