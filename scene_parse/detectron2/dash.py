"""Converts the DASH dataset into a format for detectron2 training, and trains
a detectron2 model on the dataset.
"""

import os
import json
import numpy as np
from typing import *
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


def get_dash_dicts(img_dir: str) -> List[Dict]:
    """Prepares the dataset dictionaries for the DASH dataset for detectron2
    training.

    Args:
        img_dir: The directory that the images are located.
    Returns:
        dataset_dicts: A list of dataset dictionaries, in the format:
            [
                record = {  # Information about a single image.
                    "file_name": <str>,  # The path of the example.
                    "image_id": <str or int>,  # A unique ID of the image.
                    "height": <int>,  # The height of the image.
                    "width": <int>,  # The width of the image.
                    "annotations": objs = [  # Annotations for objects in img.
                        {
                            "bbox": [  # The bbox.
                                x_min: float, 
                                y_min: float, 
                                x_max: float, 
                                y_max: float
                            ],
                            "bbox_mode": BoxMode.XYXY_ABS,  # The bbox format.
                            "category_id": <int>,  # The category label.
                            "segmentation": {  # The per-pixel seg mask, in COCO's RLE format.
                                "size": <int>,
                                "counts": <int>,
                            },
                            "iscrowd": 0,  # Docs say don't include this we don't know what this means.
                        },
                        ...
                    ]
                }
            ]
    """
    """
    Load the JSON file which contains data in the format:
        {
            k: v = {
                "filename": <str>,  # The filename of the example image.
                "regions": annos = {
                    k: anno = {
                        "region_attributes": ?,
                        "shape_attributes": anno = {
                            "all_points_x": px,
                            "all_points_y": py,

                        }
                    },
                    ...
                }
            },
            ...
        }
    """
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        # Construct the full path of the image.
        path = os.path.join(img_dir, v["filename"])

        # Retrieve the height and width of the image.
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]

            # Create a polygon datastructure with the format [(x, y)]?
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def main():
    # Tell detectron2 about the dataset function.
    for d in ["train", "val"]:
        # Associate a dataset named "dash_{split}" with the function that
        # returns the data for the split.
        DatasetCatalog.register(
            "dash_" + d, lambda d=d: get_dash_dicts("dash/" + d)
        )
        MetadataCatalog.get("dash_" + d).set(thing_classes=["dash"])
    balloon_metadata = MetadataCatalog.get("balloon_train")


if __name__ == "__main__":
    main()
