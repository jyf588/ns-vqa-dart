"""Contains the class definition for getting vision inferences."""
from argparse import Namespace
import pybullet_utils.bullet_client as bc
import torch
import torchvision.transforms as transforms

from options import BaseOptions
import dash_object


class VisionInference:
    def __init__(self, checkpoint_path: str):
        """A class for performing vision inference.

        Args:
            checkpoint_path: The path to the model checkpoint.
        """
        self.checkpoint_path = checkpoint_path

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
            gpu_ids=[0],  # Figure out how to get this automatically
            concat_img=True,
            with_depth=False,
            fp16=False,
        )
        options = BaseOptions().parse(opt=options, save_options=False)
        return options

    def predict(self) -> np.ndarray:
        """Gets a snapshot of the current scene and gets model predictions.

        Returns:
            pred: Model predictions of shape [B, N].
        """
        data = self.get_data()
        self.model.set_input(data)
        self.model.forward()
        pred = self.model.get_pred()

        return pred

    def get_data(self) -> torch.Tensor:
        """Gets the data for the current bullet scene.

        Returns:
            data: A torch tensor of size [B, C, H, W].
        """
        data = np.zeros(
            (2, self.img_height, self.img_width, 6), dtype=np.uint8
        )
        data = self.transforms(data)
        return data
