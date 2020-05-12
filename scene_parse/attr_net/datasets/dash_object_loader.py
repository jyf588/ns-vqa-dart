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

import exp.loader
from ns_vqa_dart.bullet import dash_object


class DashTorchDataset(Dataset):
    def __init__(self, exp_name: str, coordinate_frame: str, height: int, width: int):
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
        self.exp_name = exp_name
        self.coordinate_frame = coordinate_frame
        self.height = height
        self.width = width

        # Get all example indices in the experiment.
        self.idx2info = exp.loader.ExpLoader(exp_name=exp_name).get_idx2info()

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
        
        Raises:
            EOFError: If the pickle file is empty.
        """
        set_name, scene_id, timestep, oidx = self.idx2info[idx]
        scene_loader = exp.loader.SceneLoader(
            exp_name=self.exp_name, set_name=set_name, scene_id=scene_id
        )
        rgb = scene_loader.load_rgb(timestep=timestep)
        mask = scene_loader.load_mask(timestep=timestep, oidx=oidx)
        cam_dict = scene_loader.load_cam(timestep=timestep)
        odict = scene_loader.load_odict(timestep=timestep, oidx=oidx)

        X = dash_object.compute_X(img=rgb, mask=mask, keep_occluded=True)
        y = dash_object.compute_y(
            odict=odict,
            coordinate_frame=self.coordinate_frame,
            cam_position=cam_dict["position"],
            cam_orientation=cam_dict["orientation"],
        )

        X = transforms.Compose(self.normalize)(X)

        return X, y
