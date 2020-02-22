"""Generates the HTML page for visualization."""
import argparse
import os
from tqdm import tqdm
from typing import *

from bullet.dash_dataset import DashDataset
import bullet.util

START = "<html><head><style>* {font-size: 15px;}</style></head><body><table><tbody>"
END = "</tbody></table></body></html>"


def main(args: argparse.Namespace):
    dataset_dir = f"/home/michelle/datasets/{args.dataset}"

    with open(f"html/index.html", "w") as f:
        f.write(START)
        scene_captions = [f"Input RGB", "Rerendered Pred", "Mask", "Scene RGB"]
        f.write(create_caption_row(scene_captions))

        for i in tqdm(range(args.start_img_id, args.end_img_id)):
            rows = []

            oid2paths = get_object_results(dataset=args.dataset, i=i)
            oids = list(oid2paths.keys())
            first_oid = oids[0]
            later_oids = oids[1:]
            # pred_obj_paths, pred_obj_captions = get_object_results(
            #     label_type="pred", dataset=args.dataset, i=i
            # )

            rgb_path = f"datasets/{args.dataset}/rgb/{i:05}.png"
            pred_path = f"analysis/{args.dataset}/pred/{i:05}.png"
            mask_path = f"analysis/{args.dataset}/mask/{i:05}.png"

            # scene_paths = [
            #     rgb_path,
            #     pred_path,
            #     mask_path,
            #     rgb_path,
            # ] + oid2paths[first_oid]
            scene_paths = [rgb_path, pred_path, mask_path, rgb_path]

            # captions = ["" for _ in range(len(panel_labels))] + [
            #     "<br>".join(captions) for captions in obj_mask_captions
            # ]

            # f.write(get_caption_row(i, panel_labels))
            rows.append(create_img_row(paths=scene_paths))

            # for oid in later_oids:
            #     paths = [
            #         f"datasets/{args.dataset}/rgb/{i:05}.png"
            #     ] + oid2paths[oid]
            #     rows.append(create_img_row(paths=[""] * 3 + paths))

            # for obj_i in range(len(gt_obj_paths)):
            #     gt_path = gt_obj_paths[obj_i]
            #     gt_caption = "<br>".join(gt_obj_captions[obj_i])
            #     pred_path = pred_obj_paths[obj_i]
            #     pred_caption = "<br>".join(pred_obj_captions[obj_i])
            #     rows.append(create_img_row(paths=[gt_path, pred_path]))
            #     rows.append(
            #         create_caption_row(captions=[gt_caption, pred_caption])
            #     )

            for row in rows:
                f.write(row)
        f.write(END)


def create_img_row(paths: List[str]) -> str:
    """Creates an HTML row of images.

    Args:
        paths: A list of image paths to include.
    
    Returns:
        row: The HTML code for the row.
    """
    td_elems = "\n".join(
        [f'<td><img width="256" src={path}></td>' for path in paths]
    )

    row = f"""
        <tr>
            {td_elems}
        </tr>
    """
    return row


def create_caption_row(captions: List[str]) -> str:
    """Creates HTML to generate a row of captions.

    Args:
        captions: A list of captions.

    Returns:
        row: The HTML code for the row.
    """
    # td_elems = []
    # for caption in captions:
    #     td = "\n".join(caption)
    #     td_elems.append(td)

    td_row = "\n".join([f"<td>{td}</td>" for td in captions])
    row = f"""
        <tr>
            {td_row}
        </tr>
    """
    return row


def get_object_results(dataset: str, i: int):
    """Gets individual object results.

    Args:
        label_type: Either pred or gt.
        dataset: The name of the dataset.
        i: The image ID.
    
    Returns:
        paths: A list of image paths.
        captions: A list of captions corresponding to the image paths.
    
    analysis/ego_v005/20000/
        scene/
            rgb.png
            pred.png
            mask.png
        objs/
            05/
                input_rgb.png
                input_seg.png
                gt.png
                gt_caption.json
                pred.png
                pred_caption.json
    """
    oid2paths = {}
    objs_dir_rel = f"analysis/{dataset}/{i:05}/objs"
    objs_dir_abs = f"/home/michelle/analysis/{dataset}/{i:05}/objs"
    for oid in os.listdir(objs_dir_abs):
        oid_dir = os.path.join(objs_dir_rel, oid)

        oid2paths[oid] = [
            os.path.join(oid_dir, f"{n}.png")
            for n in ["input_seg", "input_rgb", "gt", "pred"]
        ]

    # paths = []
    # captions = []

    # obj_dir_name = f"{label_type}_obj"
    # obj_captions_dir_name = f"{label_type}_caption"
    # obj_masks_dir_abs = (
    #     f"/home/michelle/analysis/{dataset}/{obj_dir_name}/{i:05}"
    # )
    # obj_masks_dir_rel = f"analysis/{dataset}/{obj_dir_name}/{i:05}"
    # obj_caps_dir_abs = (
    #     f"/home/michelle/analysis/{dataset}/{obj_captions_dir_name}/{i:05}"
    # )
    # for obj_mask_fname in os.listdir(obj_masks_dir_abs):
    #     oid = int(obj_mask_fname[:-4])
    #     paths.append(os.path.join(obj_masks_dir_rel, obj_mask_fname))
    #     caption_path = os.path.join(obj_caps_dir_abs, f"{oid:02}.json")
    #     caption = bullet.util.load_json(path=caption_path)
    #     captions.append(caption)
    return oid2paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="The name of the dataset."
    )
    parser.add_argument(
        "--start_img_id", type=int, required=True, help="The start image id."
    )
    parser.add_argument(
        "--end_img_id", type=int, required=True, help="The end image id."
    )
    args = parser.parse_args()
    main(args)
