"""Contains the class definition for getting vision inferences."""
from argparse import Namespace
import imageio
import numpy as np
import os
import pybullet_utils.bullet_client as bc
import torch
import torchvision.transforms as transforms
from typing import *

from bullet import dash_object
from bullet.camera import BulletCamera
import bullet.html_images
import bullet.util
from scene_parse.attr_net.model import get_model
from scene_parse.attr_net.options import BaseOptions


class VisionInference:
    def __init__(
        self,
        p: bc.BulletClient,
        checkpoint_path: str,
        camera_position: List[float],
        camera_rotation: List[float] = [0.0, 50.0, 0.0],
        camera_offset: Optional[List[float]] = None,
        camera_directed_offset: Optional[List[float]] = None,
        img_height: int = 320,
        img_width: int = 480,
        data_height: int = 480,
        data_width: int = 480,
        coordinate_frame: Optional[str] = "camera",
        apply_offset_to_preds: Optional[bool] = None,
        assets_dir: Optional[str] = "my_pybullet_envs/assets",
        html_dir: Optional[str] = "/home/michelle/html/vision_inference",
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
            apply_offset_to_preds: Whether to apply `camera_offset` to the 
                predictions. If the vision module was trained without the 
                offset, then this should be true.
            assets_dir: The directory containing bullet assets for rerendering.
            html_dir: The directory to save HTML results in.
        """
        self.p = p
        self.checkpoint_path = checkpoint_path
        self.camera_offset = camera_offset
        self.img_height = img_height
        self.img_width = img_width
        self.data_height = data_height
        self.data_width = data_width
        self.coordinate_frame = coordinate_frame
        self.apply_offset_to_preds = apply_offset_to_preds
        self.assets_dir = assets_dir
        self.html_dir = html_dir

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

        # HTML-related settings.
        os.makedirs(html_dir, exist_ok=True)
        # Stores paths for HTML visualizations. Structure:
        # {<img_id>: <tag>: <path>}
        self.paths_dict = {}
        self.img_id = 0

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

    def predict(self, oids: List[int]) -> List[Dict]:
        """Gets a snapshot of the current scene and gets model predictions.

        Args:
            oids: A list of object IDs to get data for.

        Returns:
            odicts: A list of object dictionaries, in the order of the input
                oids.
        """
        rgb, mask = self.camera.get_rgb_and_mask()
        data = self.get_data(oids=oids, rgb=rgb, mask=mask)
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
            if self.camera_offset is not None and self.apply_offset_to_preds:
                odict["position"] = list(
                    np.array(odict["position"]) + np.array(self.camera_offset)
                )

            odict["errors"] = self.compute_object_errors(
                oid=oids[i], odict=odict
            )
            odicts.append(odict)
        self.generate_html(rgb=rgb, pred_odicts=odicts)
        self.img_id += 1
        return odicts

    def get_data(
        self, oids: List[int], rgb: np.ndarray, mask: np.ndarray
    ) -> torch.Tensor:
        """Gets the data for the current bullet scene.

        Args:
            oids: A list of object IDs to get data for.
            rgb: The RGB image.
            mask: Object-level mask.

        Returns:
            batch_data: A torch tensor of size [B, C, H, W].
        """
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

    def compute_object_errors(
        self, oid: int, odict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Computes object-level errors.

        Args:
            oid: The object ID.
            odict: The dictionary of predictions for the object.
        
        Returns:
            errors_dict: A dictionary of errors for various attributes.
        """
        errors_dict = {}
        gt_pos, gt_orn = self.p.getBasePositionAndOrientation(oid)
        errors_dict["position (cm)"] = (
            np.abs(np.array(odict["position"]) - np.array(gt_pos)) * 100
        )
        return errors_dict

    def generate_html(
        self, rgb: np.ndarray, pred_odicts: List[Dict[str, Any]]
    ):
        """Saves HTML results.
        
        Args:
            rgb: The RGB image containing the ground truth scene, that is input
                to the vision module.
            pred_odicts: A list of predicted object dictionaries to render.
        """
        # First, rerender the predictions.
        pred_objects = [
            dash_object.y_dict_to_object(odict) for odict in pred_odicts
        ]
        rerendered_pred, rerendered_pred_z = bullet.html_images.rerender(
            objects=pred_objects,
            camera=self.camera,
            assets_dir=self.assets_dir,
            check_sizes=False,
        )

        img_dir = os.path.join(self.html_dir, f"images/{self.img_id}")
        os.makedirs(img_dir, exist_ok=True)

        gt_world_path_rel = f"images/{self.img_id}/gt_world.png"
        pred_path_rel = f"images/{self.img_id}/pred.png"
        pred_z_path_rel = f"images/{self.img_id}/pred_z.png"

        gt_world_path_abs = os.path.join(self.html_dir, gt_world_path_rel)
        pred_path_abs = os.path.join(self.html_dir, pred_path_rel)
        pred_z_path_abs = os.path.join(self.html_dir, pred_z_path_rel)

        self.paths_dict[str(self.img_id)] = {
            "gt_world": gt_world_path_rel,
            "pred": pred_path_rel,
            "pred_z": pred_z_path_rel,
        }

        imageio.imwrite(gt_world_path_abs, rgb)
        imageio.imwrite(pred_path_abs, rerendered_pred)
        imageio.imwrite(pred_z_path_abs, rerendered_pred_z)

        path = os.path.join(self.html_dir, "paths.json")
        bullet.util.save_json(path=path, data=self.paths_dict)
