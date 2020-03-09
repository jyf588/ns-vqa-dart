"""Generates the HTML page for visualization.

Structure:
    <args.html_dir>/
        paths.json
        index.html
        images/
            <img_id>/
                <img_name>.png
                objs/
                    <oid>/
                        <img_name>.png
"""
import argparse
import os
from tqdm import tqdm
from typing import *

from bullet.dash_dataset import DashDataset
import bullet.util

HEADER = "<html><head><style>* {font-size: 15px;}</style></head><body><table><tbody>"
FOOTER = "</tbody></table></body></html>"


def main(args: argparse.Namespace):
    path = os.path.join(args.html_dir, "paths.json")
    img_id2paths = bullet.util.load_json(path=path)

    index_path = os.path.join(args.html_dir, "index.html")

    with open(index_path, "w") as f:
        f.write(HEADER)
        scene_tags = ["gt_world", "pred"]
        object_tags = [
            "input_seg",
            "input_rgb",
            "gt",
            "pred",
            "gt_z",
            "pred_z",
        ]
        f.write(create_caption_row(scene_tags + object_tags))

        rows = []
        for img_id, scene_paths in img_id2paths.items():
            scene_row = create_img_row(
                paths=[scene_paths[t] for t in scene_tags]
            )

            rows.append(scene_row)

            if args.show_objects:
                oid2obj_paths = scene_paths["objects"]
                for oid, obj_paths in oid2obj_paths.items():
                    rows.append(
                        create_img_row(
                            paths=[""] * len(scene_tags)
                            + [obj_paths[t] for t in object_tags]
                        )
                    )

                    captions = []
                    for cap_key in ["gt_caption", "pred_caption"]:
                        caption = bullet.util.load_json(
                            path=obj_paths[cap_key]
                        )
                        caption = "<br>".join(caption)
                        captions.append(caption)

                    rows.append(
                        create_caption_row(
                            [""] * (len(scene_tags) + 2) + captions
                        )
                    )

        for row in rows:
            f.write(row)
        f.write(FOOTER)


def create_img_row(paths: List[str]) -> str:
    """Creates an HTML row of images.

    Args:
        paths: A list of image paths to include.
    
    Returns:
        row: The HTML code for the row.
    """
    td_elems = "\n".join(
        [f'<td><img width="224" src={path}></td>' for path in paths]
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
    td_row = "\n".join([f"<td>{td}</td>" for td in captions])
    row = f"""
        <tr>
            {td_row}
        </tr>
    """
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--html_dir",
        type=str,
        required=True,
        help="The directory to run the HTML page from.",
    )
    parser.add_argument(
        "--show_objects",
        action="store_true",
        help="Whether to show object-level results.",
    )
    args = parser.parse_args()
    main(args)
