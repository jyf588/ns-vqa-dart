"""Generates the HTML page for visualization."""
import os
from tqdm import tqdm
from typing import *

from bullet.dash_dataset import DashDataset
import bullet.util

DATASET = "ego_v005"
DATASET_DIR = f"/home/michelle/datasets/{DATASET}"
START = "<html><head><style>* {font-size: 15px;}</style></head><body><table><tbody>"
END = "</tbody></table></body></html>"


def main():
    with open(f"html/index.html", "w") as f:
        f.write(START)
        for i in tqdm(range(20000, 20049)):
            obj_mask_paths, obj_mask_captions = get_obj_mask_info(i=i)

            panel_labels = [
                f"{i:05}",
                "Input RGB",
                "Mask",
                "Rerendered GT (world)",
                "Rerendered GT (camera)",
                "Rerendered Pred (camera)",
            ]
            captions = ["" for _ in range(len(panel_labels))] + [
                "<br>".join(captions) for captions in obj_mask_captions
            ]

            f.write(get_caption_row(i, panel_labels))
            f.write(get_img_row(i, obj_mask_paths))
            f.write(get_caption_row(i, captions))
        f.write(END)


def get_img_row(i: int, obj_mask_paths: List[str]) -> str:
    paths = [
        f"datasets/{DATASET}/rgb/{i}.png",
        f"analysis/{DATASET}/mask/{i}.png",
        f"analysis/{DATASET}/gt_world/{i}.png",
        f"analysis/{DATASET}/gt_cam/{i}.png",
        f"analysis/{DATASET}/pred/{i}.png",
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


def get_obj_mask_info(i: int):
    paths = []
    captions = []

    dataset = DashDataset(dataset_dir=DATASET_DIR)
    obj_masks_dir_abs = f"/home/michelle/analysis/{DATASET}/obj_masks/{i}"
    obj_masks_dir_rel = f"analysis/{DATASET}/obj_masks/{i}"
    obj_caps_dir_abs = f"/home/michelle/analysis/{DATASET}/obj_captions/{i}"
    for obj_mask_fname in os.listdir(obj_masks_dir_abs):
        oid = int(obj_mask_fname[:-4])
        # o = dataset.load_object_for_img_id_and_oid(img_id=i, oid=oid)
        # captions.append(o.to_caption())
        paths.append(os.path.join(obj_masks_dir_rel, obj_mask_fname))
        caption_path = os.path.join(obj_caps_dir_abs, f"{oid:02}.json")
        caption = bullet.util.load_json(path=caption_path)
        captions.append(caption)

    return paths, captions


if __name__ == "__main__":
    main()
