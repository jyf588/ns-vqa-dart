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
        for i in tqdm(range(args.start_img_id, args.end_img_id)):
            obj_mask_paths, obj_mask_captions = get_obj_mask_info(
                dataset=args.dataset, i=i
            )

            panel_labels = [
                f"{i:05}",
                "Input RGB",
                # "Rerendered GT (world)",
                # "Rerendered GT (camera)",
                "Rerendered Pred",
                "Mask",
            ]
            captions = ["" for _ in range(len(panel_labels))] + [
                "<br>".join(captions) for captions in obj_mask_captions
            ]

            f.write(get_caption_row(i, panel_labels))
            f.write(get_img_row(args.dataset, i, obj_mask_paths))
            f.write(get_caption_row(i, captions))
        f.write(END)


def get_img_row(dataset: str, i: int, obj_mask_paths: List[str]) -> str:
    paths = [
        f"datasets/{dataset}/rgb/{i:05}.png",
        f"analysis/{dataset}/pred/{i:05}.png",
        f"analysis/{dataset}/mask/{i:05}.png",
        # f"analysis/{dataset}/gt_world/{i:05}.png",
        # f"analysis/{dataset}/gt_cam/{i:05}.png",
    ] + obj_mask_paths

    td_elems = "\n".join(
        [f'<td><img width="256" src={path}></td>' for path in paths]
    )

    row = f"""
        <tr>
            <td>{i:05}</td>
            {td_elems}
        </tr>
    """
    return row


def get_caption_row(i: int, captions: List[str]) -> str:
    td_elems = "\n".join([f"<td>{caption}</td>" for caption in captions])

    row = f"""
        <tr>
            {td_elems}
        </tr>
    """
    return row


def get_obj_mask_info(dataset: str, i: int):
    paths = []
    captions = []

    obj_masks_dir_abs = f"/home/michelle/analysis/{dataset}/obj_masks/{i:05}"
    obj_masks_dir_rel = f"analysis/{dataset}/obj_masks/{i:05}"
    obj_caps_dir_abs = f"/home/michelle/analysis/{dataset}/obj_captions/{i:05}"
    for obj_mask_fname in os.listdir(obj_masks_dir_abs):
        oid = int(obj_mask_fname[:-4])
        paths.append(os.path.join(obj_masks_dir_rel, obj_mask_fname))
        caption_path = os.path.join(obj_caps_dir_abs, f"{oid:02}.json")
        caption = bullet.util.load_json(path=caption_path)
        captions.append(caption)

    return paths, captions


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
