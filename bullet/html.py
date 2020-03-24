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
import os
import pprint
from tqdm import tqdm
from typing import *

from ns_vqa_dart.bullet.dash_dataset import DashDataset
from ns_vqa_dart.bullet import util

HEADER = "<html><head><style>* {font-size: 15px;}</style></head><body><table><tbody>"
FOOTER = "</tbody></table></body></html>"


def main(args: argparse.Namespace):
    paths_json = os.path.join(args.html_dir, "paths.json")
    index_path = os.path.join(args.html_dir, "index.html")
    captions_path = os.path.join(args.html_dir, "captions.json")
    sid2paths = util.load_json(path=paths_json)
    captions = util.load_json(path=captions_path)

    with open(index_path, "w") as f:
        f.write(HEADER)
        scene_tags = ["rgb"]
        object_tags = ["rgb", "mask"]
        f.write(create_caption_row(scene_tags + object_tags))

        rows = []
        sids = [int(sid) for sid in sid2paths.keys()]
        for sid in sorted(sids):
            scene_paths = sid2paths[str(sid)]["scene"]
            scene_row = []
            for t in scene_tags:
                scene_row.append((scene_paths[t], "image"))
            # scene_row = create_row(paths=[scene_paths[t] for t in scene_tags])
            rows.append(create_row(row_contents=scene_row))

            if args.show_objects:
                oid2obj_paths = sid2paths[str(sid)]["objects"]
                for oid, obj_paths in oid2obj_paths.items():
                    object_row = [("", "text")] * len(scene_row)
                    for t in object_tags:
                        object_row.append((obj_paths[t], "image"))
                    for cap_tag, lines in captions[str(sid)][oid].items():
                        lines = [f"{cap_tag}:", ""] + lines
                        object_row.append((lines, "text"))
                    rows.append(create_row(row_contents=object_row))

                    # rows.append(
                    #     create_row(
                    #         paths=[""] * len(scene_tags)
                    #         + [obj_paths[t] for t in object_tags]
                    #     )
                    # )

                    # caption_elems = []
                    # for cap_tag, caption in captions[str(sid)][oid].items():
                    #     caption = "<br>".join(caption)
                    #     caption_elems.append(caption)

                    # rows.append(
                    #     create_caption_row(
                    #         [""] * (len(scene_tags) + 2) + caption_elems
                    #     )
                    # )

        for row in rows:
            f.write(row)
        f.write(FOOTER)


def create_row_elems(row_contents: Tuple[str, str]) -> List[str]:
    """
    Args:
        row_contents: A list of row contents to create HTML elements for. Each
            element is a tuple with the format (<value>, <elem_type>).
    
    Returns:
        elements: A list of HTML elements.
    """
    elements = []
    for val, elem_type in row_contents:
        if elem_type == "image":
            elem = create_img_elem(path=val)
        elif elem_type == "text":
            elem = create_text_elem(lines=val)
        else:
            raise ValueError(f"Invalid element type: {elem_type}")
        elements.append(elem)
    return elements


def create_img_elem(path: str):
    return f'<td><img width="224" src={path}></td>'


def create_text_elem(lines: List[str]):
    text = "<br>".join(lines)
    return f"<td>{text}</td>"


def create_row(row_contents: List[str]) -> str:
    """Creates an HTML row of images.

    Args:
        row_contents: A list of image paths to include.
    
    Returns:
        row: The HTML code for the row.
    """
    elements = create_row_elems(row_contents)
    td_elems = "\n".join(elements)

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
