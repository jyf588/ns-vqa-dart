"""
Generates images and captions into the following file structure:

<args.html_dir>/
    images/
        <sid>/
            <scene_tag>.png
            <oid>/
                <object_tag>.png
    paths.json

paths.json: {
    <sid>: {
        "scene": {
            <tag>: <path>
        },
        "objects": {
            <oid>: {
                <tag>: <path>
            }
        }
    }
}

"""
import argparse
import cv2
import imageio
import json
from matplotlib.pyplot import cm
import numpy as np
import os
import pprint
import random
from tqdm import tqdm
from typing import *
import sys

from ns_vqa_dart.bullet import dash_object, util
from ns_vqa_dart.bullet.camera import BulletCamera
from ns_vqa_dart.bullet.dash_dataset import DashDataset
from ns_vqa_dart.bullet.dash_object import DashObject, DashTable
from ns_vqa_dart.bullet import gen_dataset
from ns_vqa_dart.bullet.renderer import BulletRenderer
import ns_vqa_dart.bullet.camera as bullet_camera


class HTMLImageGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.paths = {}
        self.captions = {}

    def run(self):
        # Load the predictions.
        print("Loading predictions...")
        sid2info = util.load_json(path=self.args.pred_path)
        print(f"Number of total predictions: {len(sid2info)}")
        print(f"Number of scenes to visualize: {args.n_scenes}")

        # For each scene, generate scene-level and object-level images.
        # Note that sid keys are strings because they were loaded from json.
        i = 0
        tag2img = {}
        sid_strings = list(sid2info.keys())
        sorted_sample_idxs = sorted(
            random.sample(range(len(sid_strings)), args.n_scenes)
        )
        sid_strings_sampled = [sid_strings[idx] for idx in sorted_sample_idxs]
        # random.shuffle(sid_strings)
        for sid_str in tqdm(sid_strings_sampled):
            sid = int(sid_str)
            tag2img[sid] = {"scene": {}, "objects": {}}
            gt_ostates = []  # In args.coordinate_frame
            pred_ostates = []  # In args.coordinate_frame
            for oid, info in sid2info[sid_str].items():
                oid = int(oid)

                # Convert from vectors to state dictionaries.
                cam_position, cam_orientation = gen_dataset.load_camera_pose(
                    args.cam_dir,
                    sid=sid,
                    oid=oid,
                    camera_control=self.args.camera_control,
                )
                gt_state = dash_object.y_vec_to_dict(
                    y=info["labels"],
                    coordinate_frame=self.args.coordinate_frame,
                    cam_position=cam_position,
                    cam_orientation=cam_orientation,
                )
                pred_state = dash_object.y_vec_to_dict(
                    y=info["pred"],
                    coordinate_frame=self.args.coordinate_frame,
                    cam_position=cam_position,
                    cam_orientation=cam_orientation,
                )
                gt_state_untrans = dash_object.y_vec_to_dict(
                    y=info["labels"], coordinate_frame="world",
                )
                pred_state_untrans = dash_object.y_vec_to_dict(
                    y=info["pred"], coordinate_frame="world",
                )
                gt_ostates.append(gt_state)
                pred_ostates.append(pred_state)

                # Generate object-level images.
                tag2img[sid]["objects"][oid] = self.generate_object_images(
                    sid=sid, oid=oid, gt_state=gt_state, pred_state=pred_state,
                )

                # Write captions.
                self.generate_captions(
                    sid=sid,
                    oid=oid,
                    cam_position=cam_position,
                    cam_orientation=cam_orientation,
                    gt_state=gt_state,
                    pred_state=pred_state,
                    gt_state_untrans=gt_state_untrans,
                    pred_state_untrans=pred_state_untrans,
                )

            # Generate scene-level images. We do this after processing objects
            # because we need the object states.
            tag2img[sid]["scene"] = self.generate_scene_images(
                sid=sid, gt_ostates=gt_ostates, pred_ostates=pred_ostates
            )
            if i >= args.n_scenes - 1:
                break
            i += 1
        self.save_tagged_images(tag2img=tag2img)

        # Save the paths.
        util.save_json(
            path=os.path.join(self.args.html_dir, "paths.json"),
            data=self.paths,
        )

        # Save the captions.
        util.save_json(
            path=os.path.join(self.args.html_dir, "captions.json"),
            data=self.captions,
        )

    def generate_scene_images(
        self, sid: int, gt_ostates: List[Dict], pred_ostates: List[Dict]
    ) -> Dict[str, np.ndarray]:
        """Generates scene-level images.

        Args:
            sid: The scene ID.
            gt_ostates: The ground truth object states.
            pred_ostates: The predicted object states.

        Returns:
            tag2img: A dictionary with the following format:
                {
                    <tag>: <image>
                }
        """
        # third_person = gen_dataset.load_third_person_image(
        #     img_dir=self.args.img_dir, sid=sid
        # )
        # rgb, seg = gen_dataset.load_rgb_and_seg(
        #     img_dir=self.args.img_dir, sid=sid
        # )
        gt_rgb = self.rerender(states=gt_ostates, check_dims=True)
        pred_rgb = self.rerender(states=pred_ostates, check_dims=False)
        tag2img = {
            # "third_person": third_person,
            "gt": gt_rgb,
            "pred": pred_rgb,
        }
        return tag2img

    def generate_object_images(
        self,
        sid: int,
        oid: int,
        gt_state: Dict[str, Any],
        pred_state: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Generates object-level images.

        Args:
            sid: The scene ID.
            oid: The object ID.
            state: Object state with the format: 
            {
                <tag>: <path>
            }

        Returns:
            tag2img: A dictionary with the following format:
                {
                    <tag>: <image>
                }
        """
        # Given the sid and oid, load the input data.
        pickle_path = gen_dataset.sid_oid_to_pickle_path(
            dataset_dir=self.args.dataset_dir, sid=sid, oid=oid
        )
        X, _, _, _, _ = util.load_pickle(path=pickle_path)
        mask = X[:, :, :3]

        # Rerender the prediction.
        gt_rgb = self.rerender(states=[gt_state], check_dims=True)
        pred_rgb = self.rerender(states=[pred_state], check_dims=False)

        tag2img = {
            "seg": X[:, :, :3],
            "rgb": X[:, :, 3:6],
            "gt": gt_rgb,
            "pred": pred_rgb,
        }
        return tag2img

    def rerender(self, states: List[Dict], check_dims: bool):
        bc = util.create_bullet_client(mode="direct")
        renderer = BulletRenderer(p=bc)

        # Compute the orientation from the up vector because it is what the
        # renderer expects.
        for s in states:
            s["orientation"] = util.up_to_orientation(up=s["up_vector"])
            s["mass"] = 3.5
            s["mu"] = 1.0
        renderer.load_objects_from_state(
            odicts=states, position_mode="com", check_sizes=False
        )
        renderer.render_object(
            shape="tabletop",
            color="grey",
            position=[0.2, 0.2, 0.0],
            orientation=[0.0, 0.0, 0.0, 1.0],
            position_mode="com",
        )
        camera = BulletCamera(
            p=bc,
            position=[-0.2237938867122504, 0.0, 0.5425],
            rotation=[0.0, 50.0, 0.0],
            offset=[0.0, 0.2, 0.0],
        )
        rgb, _ = camera.get_rgb_and_mask()
        return rgb

    def save_tagged_images(self, tag2img: Dict):
        """
        Args:
            tag2img: Dictionary with the following format:
                {
                    <sid>: {
                        "scene": {
                            <tag>: <img>
                        },
                        "objects": {
                            <oid>: {
                                <tag>: <img>
                            }
                        }
                    }
                }
        """
        # First, create scene-level paths.
        print(f"Saving images for {len(tag2img)} scenes...")
        for sid in tqdm(tag2img.keys()):
            self.paths[sid] = {"scene": {}, "objects": {}}
            for tag, img in tag2img[sid]["scene"].items():
                rel_path = os.path.join("images", f"{sid:06}", f"{tag}.png")
                abs_path = os.path.join(self.args.html_dir, rel_path)
                self.paths[sid]["scene"][tag] = rel_path
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                imageio.imwrite(abs_path, img)

            for oid in tag2img[sid]["objects"].keys():
                assert oid not in self.paths[sid]["objects"]
                self.paths[sid]["objects"][oid] = {}
                for tag, img in tag2img[sid]["objects"][oid].items():
                    rel_path = os.path.join(
                        "images", f"{sid:06}", f"{oid:02}", f"{tag}.png"
                    )
                    abs_path = os.path.join(self.args.html_dir, rel_path)
                    self.paths[sid]["objects"][oid][tag] = rel_path
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    imageio.imwrite(abs_path, img)

    def generate_captions(
        self,
        sid: int,
        oid: int,
        cam_position: List[float],
        cam_orientation: List[float],
        gt_state: Dict,
        pred_state: Dict,
        gt_state_untrans: Dict,
        pred_state_untrans: Dict,
    ):
        """Generates captions.

        Args:
            sid: The sensor ID.
            oid: The object ID.
            gt_state: The ground truth state dictionary, in 
                args.coordinate_frame.
            pred_state: The predicted state dictionary, in 
                args.coordinate_frame.
            gt_state_untrans: The ground truth state dictionary, untransformed
                and directly containing values from y vector.
            pred_state_untrans: The predicted state dictionary, untransformed 
                and directly containing values from y vector.
        """
        state = util.load_pickle(
            path=os.path.join(self.args.states_dir, f"{sid:06}.p")
        )
        gt_state_caption = dash_object.to_caption(
            json_dict=state["objects"][oid]
        )
        gt_caption = dash_object.to_caption(json_dict=gt_state)
        pred_caption = dash_object.to_caption(json_dict=pred_state)
        gt_untrans_caption = dash_object.to_caption(json_dict=gt_state_untrans)
        pred_untrans_caption = dash_object.to_caption(
            json_dict=pred_state_untrans
        )
        if sid not in self.captions:
            self.captions[sid] = {}
        self.captions[sid][oid] = {
            "gt_state": gt_state_caption,
            f"cam_pose ({args.coordinate_frame})": [
                f"position: {cam_position}",
                f"orientation: {cam_orientation}",
            ],
            f"gt_untrans_caption ({args.coordinate_frame})": gt_untrans_caption,
            f"pred_untrans_caption ({args.coordinate_frame})": pred_untrans_caption,
            "gt_y (bullet_world)": gt_caption,
            "pred_y (bullet_world)": pred_caption,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory containing the dataset.",
    )
    parser.add_argument(
        "--states_dir",
        type=str,
        required=True,
        help="The directory containing the states.",
    )
    parser.add_argument(
        "--cam_dir",
        type=str,
        required=True,
        help="The directory containing the camera parameters.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="The path to the predictions JSON.",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="The directory containing original RGB and segmentation images.",
    )
    parser.add_argument(
        "--html_dir",
        type=str,
        required=True,
        help="The directory to run the HTML page from.",
    )
    parser.add_argument(
        "--camera_control",
        required=True,
        type=str,
        choices=["all", "center", "stack"],
        help="The method of controlling the camera.",
    )
    parser.add_argument(
        "--coordinate_frame",
        type=str,
        required=True,
        choices=["world", "camera", "unity_camera"],
        help="The coordinate frame that predictions are in.",
    )
    parser.add_argument(
        "--n_scenes",
        type=int,
        help="Number of example scenes to generate images for.",
    )
    args = parser.parse_args()
    generator = HTMLImageGenerator(args=args)
    generator.run()
