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
        print(f"Number of objects to visualize: {args.n_objects}")

        # For each scene, generate scene-level and object-level images.
        # Note that sid keys are strings because they were loaded from json.
        i = 0
        tag2img = {}
        sid_strings = list(sid2info.keys())
        random.shuffle(sid_strings)
        for sid_str in tqdm(sid_strings):
            sid = int(sid_str)
            tag2img[sid] = {"scene": {}, "objects": {}}
            gt_ostates = []
            pred_ostates = []
            for oid, info in sid2info[sid_str].items():
                oid = int(oid)

                # Convert from vectors to state dictionaries.
                gt_state = dash_object.y_vec_to_dict(
                    y=info["labels"],
                    coordinate_frame=self.args.coordinate_frame,
                )
                pred_state = dash_object.y_vec_to_dict(
                    y=info["pred"], coordinate_frame=self.args.coordinate_frame
                )
                gt_ostates.append(gt_state)
                pred_ostates.append(pred_state)

                # Generate object-level images.
                tag2img[sid]["objects"][oid] = self.generate_object_images(
                    sid=sid, oid=oid, gt_state=gt_state, pred_state=pred_state
                )

                # Write captions.
                self.generate_captions(
                    sid=sid, oid=oid, gt_state=gt_state, pred_state=pred_state
                )

            # Generate scene-level images. We do this after processing objects
            # because we need the object states.
            # tag2img[sid]["scene"] = self.generate_scene_images(
            #     sid=sid, gt_ostates=gt_ostates, pred_ostates=pred_ostates
            # )
            if i > args.n_objects:
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

        Returns:
            tag2img: A dictionary with the following format:
                {
                    <tag>: <image>
                }
        """
        rgb, seg = gen_dataset.load_rgb_and_seg(
            img_dir=self.args.img_dir, sid=sid
        )
        gt_rgb = self.rerender(states=gt_ostates)
        pred_rgb = self.rerender(states=pred_ostates)
        tag2img = {"rgb": rgb, "seg": seg, "gt": gt_rgb, "pred": pred_rgb}
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
        renderer.load_objects_from_state(
            ostates=states, position_mode="com", check_sizes=False
        )
        renderer.render_object(
            o=DashTable(position=[0.2, 0.2, 0.0]), position_mode="com"
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
        self, sid: int, oid: int, gt_state: Dict, pred_state: Dict
    ):
        gt_caption = dash_object.to_caption(json_dict=gt_state)
        pred_caption = dash_object.to_caption(json_dict=pred_state)
        if sid not in self.captions:
            self.captions[sid] = {}
        self.captions[sid][oid] = {"gt_y": gt_caption, "pred_y": pred_caption}


def main(args: argparse.Namespace):

    print(f"Loading objects...")
    img_id2paths = {}
    for out_dict in tqdm(out_dicts):
        path = out_dict["path"]
        pred = out_dict["pred"]
        sid, oid = gen_dataset.path_to_sid_oid(path=path)
        X, y = util.load_pickle(path=path)

        # Convert from labels from vector to dictionary format.
        pred_dict = dash_object.y_vec_to_dict(
            y=pred, coordinate_frame=args.coordinate_frame
        )
        gt_dict = dash_object.y_vec_to_dict(
            y=list(y), coordinate_frame=args.coordinate_frame
        )

        tag2img = {"mask": X[:, :, :3], "rgb": X[:, :, 3:6]}

        rel_img_dir = os.path.join("images", f"{sid:06}")
        abs_img_dir = os.path.join(args.html_dir, rel_img_dir)
        os.makedirs(abs_img_dir, exist_ok=True)

        # Write the scene-level information.
        for tag, img in tag2img.items():
            rel_path = os.path.join(rel_img_dir, f"{tag}.png")
            abs_path = os.path.join(abs_img_dir, f"{tag}.png")
            imageio.imwrite(abs_path, img)
            if sid not in img_id2paths:
                img_id2paths[sid] = {}
            img_id2paths[sid][tag] = rel_path

        # Convert dictionary to object format.
        # o = dash_object.y_dict_to_object(
        #     y_dict=y_dict,
        #     img_id=img_id,
        #     oid=oid,
        #     gt_orientation=gt_o.orientation,
        # )
        # if img_id not in img_id2oid2pred_object:
        #     img_id2oid2pred_object[img_id] = {}
        # img_id2oid2pred_object[img_id][oid] = o

    # Save the paths.
    path = os.path.join(args.html_dir, "paths.json")
    util.save_json(path=path, data=img_id2paths)
    return


def rerender(
    objects: List[DashObject],
    camera: BulletCamera,
    # assets_dir: Optional[str] = "bullet/assets",
    check_sizes: Optional[bool] = True,
) -> np.ndarray:
    """Rerenders a scene given a list of DashObjects generated by a 
    DashDataset.

    Args:
        objects: A list of DashObjects in the scene.
        camera: The camera of the scene.
        assets_dir: The directory containing bullet assets.
        check_sizes: Whether to check the object sizes.

    Returns:
        rerendered: The rerendered image.
        rerendered_z: The rerendered image aligned with z axis.
    """
    p = util.create_bullet_client(mode="direct")
    renderer = BulletRenderer(p=p)

    [
        renderer.render_object(
            o=o, position_mode="com", check_sizes=check_sizes
        )
        for o in objects + [DashTable(position=[0.2, 0.2, 0.0])]
    ]

    # We create a copy because we don't want to override the bullet client of
    # the input camera.
    rerender_cam = bullet_camera.from_json(camera.to_json())
    rerender_cam.set_bullet_client(p=p)
    rerendered, _ = rerender_cam.get_rgb_and_mask()

    # Create a camera just to get the z axis view.
    rerendered_z, _ = BulletCamera(p=p, init_type="z_axis").get_rgb_and_mask()

    return rerendered, rerendered_z


def world2cam2world(
    world_objects: List[DashObject], camera: BulletCamera
) -> List[DashObject]:
    cam_objects = []
    for o in world_objects:
        world_position = o.position
        world_up_vector = o.compute_up_vector()

        cam_position = util.world_to_cam(xyz=world_position, camera=camera)
        cam_up_vector = util.world_to_cam(xyz=world_up_vector, camera=camera)

        world_position = util.cam_to_world(xyz=cam_position, camera=camera)
        world_up_vector = util.cam_to_world(xyz=cam_up_vector, camera=camera)

        o.position = world_position
        o.up_vector = world_up_vector

        cam_objects.append(o)
    return cam_objects


def convert_mask_to_img(mask: np.ndarray):
    H, W = mask.shape
    mask_img = np.zeros((H, W, 3)).astype(np.uint8)
    oids = np.unique(mask)
    color = iter(cm.rainbow(np.linspace(0, 1, len(oids))))
    for oid in oids:
        mask_img[mask == oid] = next(color)[:3] * 255
    return mask_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="The directory containing the dataset.",
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
        "--coordinate_frame",
        type=str,
        required=True,
        choices=["world", "camera"],
        help="The coordinate frame that predictions are in.",
    )
    parser.add_argument(
        "--n_objects",
        type=int,
        help="Number of example objects to generate images for.",
    )
    args = parser.parse_args()
    generator = HTMLImageGenerator(args=args)
    generator.run()
