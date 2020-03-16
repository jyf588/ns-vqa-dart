"""Contains the class definition for getting vision inferences."""
from argparse import Namespace
import imageio
import numpy as np
import os
import pybullet_utils.bullet_client as bc
import torch
import torchvision.transforms as transforms
from typing import *
import sys

from ns_vqa_dart.bullet import dash_object
from ns_vqa_dart.bullet import html_images
from ns_vqa_dart.bullet import util
from ns_vqa_dart.bullet.camera import BulletCamera
from ns_vqa_dart.bullet.dash_object import DashTable, DashRobot
from ns_vqa_dart.bullet.renderer import BulletRenderer
from ns_vqa_dart.bullet.state_saver import StateSaver
from ns_vqa_dart.scene_parse.attr_net.model import get_model
from ns_vqa_dart.scene_parse.attr_net.options import BaseOptions


class VisionInference:
    def __init__(
        self,
        # p: bc.BulletClient,
        state_saver: StateSaver,
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
        html_dir: Optional[str] = None,
        save_html_freq: Optional[int] = 5,
    ):
        """A class for performing vision inference.

        Args:
            p: The bullet client to use, provided by the client.
            state_saver: Tracks the state of the client scene.
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
            html_dir: The directory to save HTML results in.
        """
        # self.client_p = p
        self.state_saver = state_saver
        self.checkpoint_path = checkpoint_path
        self.camera_offset = camera_offset
        self.img_height = img_height
        self.img_width = img_width
        self.data_height = data_height
        self.data_width = data_width
        self.coordinate_frame = coordinate_frame
        self.apply_offset_to_preds = apply_offset_to_preds
        self.html_dir = html_dir
        self.save_html_freq = save_html_freq

        # Create own bullet client for reconstructing.
        self.vision_p = util.create_bullet_client(mode="direct")

        # Initialize renderer.
        self.renderer = BulletRenderer(p=self.vision_p)

        self.robot = DashRobot(
            p=self.vision_p,
            urdf_path="my_pybullet_envs/assets/inmoov_ros/inmoov_description/robots/inmoov_shadow_hand_v2_1_revolute_head.urdf",
            position=[-0.3, 0.5, -1.25],
        )

        # Render tabletop.
        _ = self.renderer.render_object(
            o=DashTable(position=[0.2, 0.2, 0.0]), position_mode="com"
        )

        # Camera initialization.
        self.camera = BulletCamera(
            p=self.vision_p,
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
        if self.html_dir is not None:
            os.makedirs(html_dir, exist_ok=True)
        # Stores paths for HTML visualizations. Structure:
        # {<img_id>: <tag>: <path>}
        self.paths_dict = {}
        self.img_id = 0

    def close(self):
        self.vision_p.disconnect()

    def get_options(self):
        """Creates the options namespace to define the vision model."""
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

    def predict(self, client_oids: List[int]) -> List[Dict]:
        """Reconstructs the client's scene and retrieves model predictions.

        Args:
            oids: A list of object IDs to make predictions for.

        Returns:
            odicts: A list of object dictionaries, in the order of the input
                oids.
        """
        # 1. Get the current state of the client scene.
        state = self.state_saver.get_current_state()

        # 2. Load objects with their states.
        vision_oids = self.renderer.load_objects_from_state(
            ostates=state["objects"], position_mode="com"
        )

        # 3. Update robot state.
        self.robot.set_state(state=state["robot"])

        # Convert from client's oids to predict into vision oids, since the
        # reconstructed vision scene could have different oids.
        client2vision_oids = {}
        for i in range(len(state["objects"])):
            client_oid = state["objects"][i]["oid"]
            vision_oid = vision_oids[i]
            client2vision_oids[client_oid] = vision_oid

        oids_to_predict = [
            client2vision_oids[client_oid] for client_oid in client_oids
        ]

        # Get a snapshot of the scene.
        rgb, mask = self.camera.get_rgb_and_mask()

        # 3. Create the input data to the model from the scene, and run forward
        # prediction.
        pred = self.predict_scene(oids=oids_to_predict, rgb=rgb, mask=mask)

        # Convert vector predictions into dictionary format.
        odicts = self.process_predictions(pred=pred)

        # Optionally generate visuals.
        if (
            self.html_dir is not None
            and self.img_id % self.save_html_freq == 0
        ):
            self.generate_html(rgb=rgb, pred_odicts=odicts)

        self.img_id += 1

        self.renderer.remove_objects(ids=vision_oids)

        return odicts

    def predict_scene(self, oids: List[int], rgb, mask) -> np.ndarray:
        """From the current scenes, constructs input data and runs forward pass
        through the model to retrieve predictions.

        Args:
            oids: A list of object IDs to make predictions for.
        
        Returns:
            pred: A vector format of model output predictions.
        """
        data = self.get_data(oids=oids, rgb=rgb, mask=mask)
        self.model.set_input(data)
        self.model.forward()
        pred = self.model.get_pred()
        return pred

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

    def process_predictions(self, pred: np.ndarray) -> List[Dict]:
        """Converts vector predictions output by the model into an 
        interpretable dictionary of object predictions. Additional things 
        include:
            1. Applying an offset to predictions (e.g., if the camera during 
                test time is offset compared to the camera during training)
            2. Computing object prediction errors.
        
        Args:
            pred: A vector format of model output predictions.

        Returns:
            odicts: A list of dictionaries containing object predictions.
        """
        odicts = []
        for i in range(len(pred)):
            odict = dash_object.y_vec_to_dict(
                y=list(pred[i]),
                coordinate_frame=self.coordinate_frame,
                camera=self.camera,
            )

            # Apply the camera offset to predictions if specified.
            if self.camera_offset is not None and self.apply_offset_to_preds:
                odict["position"] = list(
                    np.array(odict["position"]) + np.array(self.camera_offset)
                )

            # odict["errors"] = self.compute_object_errors(
            #     oid=odict["oid"], odict=odict
            # )
            odicts.append(odict)
        return odicts

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
        rerendered_pred, rerendered_pred_z = html_images.rerender(
            objects=pred_objects, camera=self.camera, check_sizes=False
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
        util.save_json(path=path, data=self.paths_dict)
