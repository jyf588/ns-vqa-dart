import os
import cv2
import json
import random
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


def get_balloon_dicts(img_dir):
    """Prepares the dataset dictionaries for the balloon dataset for detectron2
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
                            "segmentation": [poly],  # The poly segmentation.
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

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
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
    # Register the dataset for each split.
    for d in ["train", "val"]:
        # Associate `ballon_{split}` with the corresponding function that
        # returns the data for the split.
        DatasetCatalog.register(
            "balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d)
        )
        # Add the `thing_classes` metadata, which is used by all instance
        # detection / segmentation tasks. It stores a list of names for each
        # instance / thing category.
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

    # Verify that the dataloading is correct.
    # balloon_metadata = MetadataCatalog.get("balloon_train")
    # dataset_dicts = get_balloon_dicts("balloon/train")
    # for idx, d in enumerate(random.sample(dataset_dicts, 3)):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(
    #         img[:, :, ::-1], metadata=balloon_metadata, scale=0.5
    #     )
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imwrite(f"balloon_train_{idx}.png", vis.get_image()[:, :, ::-1])

    # Train.
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128  # faster, and good enough for this toy dataset (default: 512)
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # Validate.
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    trainer.resume_or_load(resume=True)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.7  # set the testing threshold for this model
    )
    cfg.DATASETS.TEST = ("balloon_val",)
    predictor = DefaultPredictor(cfg)

    # Visualize val results.
    dataset_dicts = get_balloon_dicts("balloon/val")
    for idx, d in enumerate(random.sample(dataset_dicts, 3)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=balloon_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(f"balloon_val_{idx}.png", v.get_image()[:, :, ::-1])

    # Evaluate performance.
    evaluator = COCOEvaluator(
        "balloon_val", cfg, False, output_dir="./output/"
    )
    val_loader = build_detection_test_loader(cfg, "balloon_val")
    res = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(res)


if __name__ == "__main__":
    main()
