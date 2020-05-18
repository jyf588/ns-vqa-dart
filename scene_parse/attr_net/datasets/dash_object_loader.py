import os
import cv2
import sys
import time
import json
import pprint
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
    def __init__(self, data_dirs: List[str], split: str, split_frac=0.8):
        """A Pytorch Dataset for DASH objects.

        Args:
            data_dirs: A list of data directories to load data from. Each directory 
            should be a folder of pickle files.
        """
        print(f"*****Initializing Dataset*****")
        self.paths = []

        # Loop over the directories.
        for data_dir in data_dirs:
            print(f"Gathering data from {data_dir}...")
            p = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
            split_paths = util.compute_split(split, p, split_frac)
            self.paths += split_paths

            print(f"First 5 examples selected:")
            pprint.pprint(split_paths[:5])

        self.normalize = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
        ]

        print(
            f"Initialized DashTorchDataset for data_dir {data_dir}, split {split} containing {len(self)} examples."
        )

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
        try:
            data = util.load_pickle(path)
        except:
            print(
                f"Warning: EOF error when reading pickle file {path} for idx {idx}. Sampling new example."
            )
            # Regenerate idxs until we get successful loading.
            retries = 0
            while retries < 50:
                retries += 1
                idx = random.randint(0, self.__len__())
                path = self.paths[idx]
                try:
                    data = util.load_pickle(path)
                    break
                except:
                    print(
                        f"Warning: EOF error when reading pickle file {path} for idx {idx}. Sampling new example. Retries: {retries}"
                    )

        X_before_normalize, y, sid, oid, path = data
        X = transforms.Compose(self.normalize)(X_before_normalize)

        # input_rgb = np.hstack(
        #     [X_before_normalize[:, :, :3], X_before_normalize[:, :, 3:6]]
        # )
        # cv2.imshow("example", input_rgb[:, :, ::-1])

        # normalized_rgb = X.numpy()
        # normalized_rgb = np.moveaxis(normalized_rgb, 0, -1)
        # normalized_rgb = np.hstack(
        #     [normalized_rgb[:, :, :3], normalized_rgb[:, :, 3:6]]
        # )
        # bgr_normalized_img = normalized_rgb[:, :, ::-1]
        # cv2.imshow("normalized", bgr_normalized_img)
        # cv2.waitKey(0)

        return X, y, sid

    def load_example(self, idx):
        path = self.idx2path(idx)
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data, path
