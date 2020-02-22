"""Visualizes images and labels for a particular dataset."""
import argparse
import cv2
import imageio
import json
from matplotlib.pyplot import cm
import numpy as np
import os
from tqdm import tqdm
from typing import *

from bullet.camera import BulletCamera
from bullet.dash_dataset import DashDataset
import bullet.dash_object
from bullet.dash_object import DashObject, DashTable
from bullet.renderer import BulletRenderer
import bullet.util


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    # Create an instance of a DashDataset.
    dataset = DashDataset(dataset_dir=args.dataset_dir)

    # Load the predictions.
    print("Loading predictions...")
    pred_dicts = bullet.util.load_json(path=args.pred_path)
    print(f"Number of total predictions: {len(pred_dicts)}")
    print(f"Number of objects to visualize: {args.n_examples}")
    img_id2oid2pred_object = {}
    if args.n_examples is not None:
        pred_dicts = pred_dicts[: args.n_examples]
    for pred_dict in tqdm(pred_dicts):
        img_id = pred_dict["img_id"]
        oid = pred_dict["oid"]
        y = pred_dict["pred"]

        camera = dataset.load_camera_for_eid(eid=img_id)
        y_dict = bullet.dash_object.y_vec_to_dict(
            y=y, coordinate_frame=args.coordinate_frame, camera=camera
        )
        gt_o = dataset.load_object_for_img_id_and_oid(img_id=img_id, oid=oid)
        o = bullet.dash_object.y_dict_to_object(
            y_dict=y_dict,
            img_id=img_id,
            oid=oid,
            gt_orientation=gt_o.orientation,
        )
        if img_id not in img_id2oid2pred_object:
            img_id2oid2pred_object[img_id] = {}
        img_id2oid2pred_object[img_id][oid] = o

    pred_img_ids = list(img_id2oid2pred_object.keys())

    # For each example, load the rgb image and mask.
    for img_id in tqdm(pred_img_ids):
        gt_objects_world, camera, rgb, mask = dataset.load_example(eid=img_id)

        # Convert the mask to an image.
        mask_img = convert_mask_to_img(mask=mask)

        # Rerender the scene from the GT labels.
        rerendered_gt_world = rerender(objects=gt_objects_world, camera=camera)

        # Rerender GT using world -> cam -> world.
        gt_objects_cam = world2cam2world(
            world_objects=gt_objects_world, camera=camera
        )
        rerendered_gt_cam = rerender(objects=gt_objects_cam, camera=camera)

        # Rerender the scene from the model predictions.
        oid2pred_object = img_id2oid2pred_object[img_id]
        pred_objects = list(oid2pred_object.values())
        rerendered_pred = rerender(
            objects=pred_objects, camera=camera, check_sizes=False
        )

        name2img = {
            "mask": mask_img,
            "gt_world": rerendered_gt_world,
            "gt_cam": rerendered_gt_cam,
            "pred": rerendered_pred,
        }

        # Write the scene-level information.
        for k, img in name2img.items():
            img_dir = os.path.join(args.output_dir, k)
            os.makedirs(img_dir, exist_ok=True)
            path = os.path.join(img_dir, f"{img_id:05}.png")
            imageio.imwrite(path, img)

        # Create object masks.
        oid2gt_objects_world = {o.oid: o for o in gt_objects_world}
        for oid in oid2pred_object.keys():
            pred_o = oid2pred_object[oid]
            gt_o = oid2gt_objects_world[oid]
            data = dataset.compute_data_from_rgb_and_mask(
                o=gt_o, rgb=rgb, mask=mask
            )

            input_seg = data[:, :, :3]
            input_rgb = data[:, :, 3:6]

            save_obj_results(
                oid=oid,
                img_id=img_id,
                input_seg=input_seg,
                # new_seg=new_seg,
                # seg_black=seg_black,
                input_rgb=input_rgb,
                # input_rgb_seg=input_rgb_seg,
                gt_o=gt_o,
                pred_o=pred_o,
                camera=camera,
            )


def save_obj_results(
    oid: int,
    img_id: int,
    input_seg: np.ndarray,
    # new_seg: np.ndarray,
    # seg_black: np.ndarray,
    input_rgb: np.ndarray,
    # input_rgb_seg: np.ndarray,
    gt_o: DashObject,
    pred_o: DashObject,
    camera: BulletCamera,
):
    gt_img = rerender(objects=[gt_o], camera=camera, check_sizes=False)
    pred_img = rerender(objects=[pred_o], camera=camera, check_sizes=False)

    obj_dir = os.path.join(
        args.output_dir, f"{img_id:05}", "objs", f"{oid:02}"
    )
    os.makedirs(obj_dir, exist_ok=True)

    imageio.imwrite(os.path.join(obj_dir, f"input_seg.png"), input_seg)
    # imageio.imwrite(os.path.join(obj_dir, f"new_seg.png"), new_seg)
    # imageio.imwrite(os.path.join(obj_dir, f"seg_black.png"), seg_black)
    imageio.imwrite(os.path.join(obj_dir, f"input_rgb.png"), input_rgb)
    # imageio.imwrite(os.path.join(obj_dir, f"input_rgb_seg.png"), input_rgb_seg)
    imageio.imwrite(os.path.join(obj_dir, f"gt.png"), gt_img)
    imageio.imwrite(os.path.join(obj_dir, f"pred.png"), pred_img)

    # gt_cap_path = os.path.join(gt_cap_dir, f"{oid:02}.json")
    # pred_cap_path = os.path.join(pred_cap_dir, f"{oid:02}.json")
    # bullet.util.save_json(path=gt_cap_path, data=gt_o.to_caption())
    # bullet.util.save_json(path=pred_cap_path, data=pred_o.to_caption())


def create_visual(
    name2img: Dict[str, np.ndarray], name2caption_text: Dict[str, str]
) -> np.ndarray:
    """Applies captions to each image, and combines into a single image.

    Args:
        name2img: A dictionary mapping from the name of each image to the image
            itself.
    
    Returns:
        visual: The final visual containing captions on all images combined.
    """
    name2captioned_img = {}
    max_img_height = 0
    line_height = 27
    for k, img in name2img.items():
        lines = name2caption_text[k]
        H, W, _ = img.shape
        caption = np.ones((500, W, 3)).astype(np.uint8) * 255
        for line_i, line in enumerate(lines):
            caption = cv2.putText(
                caption,
                str(line),
                (5, 5 + line_height * (line_i + 1)),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.7,
                color=(0, 0, 0),
            )
        captioned_img = np.vstack([img, caption])
        name2captioned_img[k] = captioned_img
        H = captioned_img.shape[0]
        if H > max_img_height:
            max_img_height = H

    visual = np.hstack(list(name2captioned_img.values()))
    return visual


def rerender(
    objects: List[DashObject],
    camera: BulletCamera,
    check_sizes: Optional[bool] = True,
) -> np.ndarray:
    """Rerenders a scene given a JSON dictionary of labels generated by a 
    DashDataset.

    Args:
        objects: A list of DashObjects in the scene.
        camera: The camera of the scene.
        check_sizes: Whether to check the object sizes.

    Returns:
        rerendered: The rerendered image.
    """
    p = bullet.util.create_bullet_client(mode="direct")
    renderer = BulletRenderer(p=p)
    objects_to_render = objects
    renderer.render_table(DashTable())
    [
        renderer.render_object(o=o, check_sizes=check_sizes)
        for o in objects_to_render
    ]
    rerendered, _ = camera.get_rgb_and_mask(p=p)
    return rerendered


def world2cam2world(
    world_objects: List[DashObject], camera: BulletCamera
) -> List[DashObject]:
    cam_objects = []
    for o in world_objects:
        world_position = o.position
        world_up_vector = o.compute_up_vector()

        cam_position = bullet.util.world_to_cam(
            xyz=world_position, camera=camera
        )
        cam_up_vector = bullet.util.world_to_cam(
            xyz=world_up_vector, camera=camera
        )

        world_position = bullet.util.cam_to_world(
            xyz=cam_position, camera=camera
        )
        world_up_vector = bullet.util.cam_to_world(
            xyz=cam_up_vector, camera=camera
        )

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
        "--output_dir",
        type=str,
        required=True,
        help="The directory to save the visualizations.",
    )
    parser.add_argument(
        "--coordinate_frame",
        type=str,
        required=True,
        choices=["world", "camera"],
        help="The coordinate frame that predictions are in.",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        help="Number of examples to generate images for.",
    )
    args = parser.parse_args()
    main(args)
