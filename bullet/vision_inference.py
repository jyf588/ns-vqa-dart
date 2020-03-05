"""Contains the class definition for getting vision inferences."""
from argparse import Namespace
import numpy as np
import pybullet_utils.bullet_client as bc
import torch
import torchvision.transforms as transforms
from typing import *

from bullet import dash_object
from scene_parse.attr_net.model import get_model
from scene_parse.attr_net.options import BaseOptions


class VisionInference:
    def __init__(
        self,
        checkpoint_path: str,
        img_height: int = 320,
        img_width: int = 480,
        coordinate_frame: Optional[str] = "world",
    ):
        """A class for performing vision inference.

        Args:
            checkpoint_path: The path to the model checkpoint.
            img_height: The height of the image.
            img_width: The width of the image.
            coordinate_frame: The coordinate frame the model predictions are 
                in.
        """
        self.checkpoint_path = checkpoint_path
        self.img_height = img_height
        self.img_width = img_width
        self.coordinate_frame = coordinate_frame

        options = self.get_options()
        self.model = get_model(options)
        self.model.eval_mode()

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
            ]
        )

    def get_options(self):
        """Creates the options namespace to define the vision model.
        """
        options = Namespace(
            inference_only=True,
            load_checkpoint_path=self.checkpoint_path,
            gpu_ids="0",
            concat_img=True,
            with_depth=False,
            fp16=False,
        )
        options = BaseOptions().parse(opt=options, save_options=False)
        return options

    def predict(self) -> np.ndarray:
        """Gets a snapshot of the current scene and gets model predictions.

        Returns:
            odicts: A list of object dictionaries.
        """
        data = self.get_data()
        self.model.set_input(data)
        self.model.forward()
        pred = self.model.get_pred()

        odicts = []
        for i in range(len(pred)):
            odict = dash_object.y_vec_to_dict(
                y=pred[i], coordinate_frame=self.coordinate_frame
            )
            odicts.append(odict)
        return odicts

    def get_data(self) -> torch.Tensor:
        """Gets the data for the current bullet scene.

        Returns:
            batch_data: A torch tensor of size [B, C, H, W].
        """
        n_objects = 4
        batch_data = torch.zeros(
            size=(n_objects, 6, self.img_height, self.img_width)
        )
        for i in range(n_objects):
            data = np.zeros(
                (self.img_height, self.img_width, 6), dtype=np.uint8
            )
            batch_data[i] = self.transforms(data)
        return batch_data
