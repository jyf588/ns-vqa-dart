"""Contains the class definition for getting vision inferences."""
from argparse import Namespace
import numpy as np
import pybullet_utils.bullet_client as bc
import torch
import torchvision.transforms as transforms
from typing import *

from bullet.camera import BulletCamera
from bullet import dash_object
from scene_parse.attr_net.model import get_model
from scene_parse.attr_net.options import BaseOptions


class VisionInference:
    def __init__(
        self,
        p: bc.BulletClient,
        checkpoint_path: str,
        camera_position: List[float] = [  # Robot head position
            -0.2237938867122504,
            0.03198004185028341,
            0.5425,
        ],
        camera_rotation: List[float] = [0.0, 50.0, 0.0],
        camera_offset: Optional[List[float]] = None,
        camera_directed_offset: Optional[List[float]] = [0.02, 0.0, 0.0],
        img_height: int = 320,
        img_width: int = 480,
        data_height: int = 480,
        data_width: int = 480,
        coordinate_frame: Optional[str] = "camera",
    ):
        """A class for performing vision inference.

        Args:
            p: The bullet client to use.
            checkpoint_path: The path to the model checkpoint.
            camera_position: The position of the camera.
            camera_rotation: The roll, pitch, and yaw of the camera (degrees).
            camera_offset: The amount to offset the camera position in order to
                match the camera position that the model was trained on.
            camera_directed_offset: The position offset to apply in the direction of 
                the camera forward vector.
            img_height: The height of the image.
            img_width: The width of the image.
            data_height: The height of the input data to the model.
            data_width: The width of the input data to the model.
            coordinate_frame: The coordinate frame the model predictions are 
                in.
        """
        self.p = p
        self.checkpoint_path = checkpoint_path
        self.camera_offset = camera_offset
        self.img_height = img_height
        self.img_width = img_width
        self.data_height = data_height
        self.data_width = data_width
        self.coordinate_frame = coordinate_frame

        # Camera initialization.
        self.camera = BulletCamera(
            p=p,
            position=camera_position,
            rotation=camera_rotation,
            offset=camera_offset,
            directed_offset=camera_directed_offset,
        )

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

    def predict(self, oids: List[int]) -> np.ndarray:
        """Gets a snapshot of the current scene and gets model predictions.

        Args:
            oids: A list of object IDs to get data for.

        Returns:
            odicts: A list of object dictionaries.
        """
        data = self.get_data(oids=oids)
        self.model.set_input(data)
        self.model.forward()
        pred = self.model.get_pred()

        odicts = []
        for i in range(len(pred)):
            odict = dash_object.y_vec_to_dict(
                y=list(pred[i]),
                coordinate_frame=self.coordinate_frame,
                camera=self.camera,
            )

            # Apply the camera offset.
            if self.camera_offset is not None:
                odict["position"] = list(
                    np.array(odict["position"]) + np.array(self.camera_offset)
                )
            odicts.append(odict)
        return odicts

    def get_data(self, oids: List[int]) -> torch.Tensor:
        """Gets the data for the current bullet scene.

        Args:
            oids: A list of object IDs to get data for.

        Returns:
            batch_data: A torch tensor of size [B, C, H, W].
        """
        rgb, mask = self.camera.get_rgb_and_mask()
        batch_data = torch.zeros(
            size=(len(oids), 6, self.data_height, self.data_width)
        )
        for i, oid in enumerate(oids):
            data = dash_object.compute_data_from_rgb_and_mask(
                oid=oid,
                rgb=rgb,
                mask=mask,
                data_height=self.data_height,
                data_width=self.data_width,
            )
            batch_data[i] = self.transforms(data)
        return batch_data
