"""Contains the class definition for getting vision inferences."""
import pybullet_utils.bullet_client as bc
import torch
import torchvision.transforms as transforms


class VisionInference:
    def __init__(self, p: bc.BulletClient):
        """A class for performing vision inference.

        Args:
            p: The bullet client to use to obtain scene data.
        """
        opt = get_options("test")
        self.model = get_model(opt)
        self.model.eval_mode()

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
            ]
        )

    def predict(self):
        data = self.get_data()
        self.model.set_input(data)
        self.model.forward()
        pred = self.model.get_pred()

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
