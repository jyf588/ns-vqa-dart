"""Converts the DASH dataset into a format for detectron2 training, and trains
a detectron2 model on the dataset.
"""
import os
import cv2
import json
import pickle
import random
import imageio
import numpy as np
from typing import *
import pycocotools.mask

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

from ns_vqa_dart.bullet.seg import seg_img_to_map


SHAPE2INT = {
    "box": 0,
    "cylinder": 1,
    "sphere": 2,
}


def get_dash_dicts(split: str) -> List[Dict]:
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
    dataset_dicts = []
    if split == "train":
        start_idx = 0
        end_idx = 80
    elif split == "val":
        start_idx = 0
        end_idx = 1
    for idx in range(start_idx, end_idx):
        record = {}

        # Construct the full path of the image.
        # path = os.path.join(img_dir, v["filename"])
        state_path = (
            f"/media/sdc3/mguo/data/states/full/planning_v003_20K/{idx:06}.p"
        )
        rgb_path = f"/media/sdc3/mguo/data/datasets/planning_v003_20K/unity_output/images/first/rgb/{idx:06}_0.png"
        seg_path = f"/media/sdc3/mguo/data/datasets/planning_v003_20K/unity_output/images/first/seg/{idx:06}_0.png"

        state = pickle.load(open(state_path, "rb"))

        # Retrieve the height and width of the image.
        seg_img = imageio.imread(seg_path)
        height, width = imageio.imread(rgb_path).shape[:2]

        record["file_name"] = rgb_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # Compute the segmentations.
        seg_map, oids = seg_img_to_map(seg_img=seg_img)

        objs = []
        for oid, odict in state["objects"].items():
            mask = seg_map == oid
            rle = pycocotools.mask.encode(np.asarray(mask, order="F"))

            # Convert `counts` to ascii, otherwise json dump complains about 
            # not being able to serialize bytes.
            # https://github.com/facebookresearch/detectron2/issues/200#issuecomment-614407341
            rle["counts"] = rle["counts"].decode("ASCII")
            assert isinstance(rle, dict)
            assert len(rle) > 0
            bbox = list(pycocotools.mask.toBbox(rle))
            obj = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": rle,
                "category_id": SHAPE2INT[odict["shape"]],
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
            "dash_" + d, lambda d=d: get_dash_dicts(split=d)
        )
        MetadataCatalog.get("dash_" + d).set(
            thing_classes=list(SHAPE2INT.keys())
        )

    # Verify that the dataloading is correct.
    dash_metadata = MetadataCatalog.get("dash_train")
    dataset_dicts = get_dash_dicts(split="train")
    for idx, d in enumerate(random.sample(dataset_dicts, 3)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(
            img[:, :, ::-1], metadata=dash_metadata, scale=0.5
        )
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(
            f"ns_vqa_dart/scene_parse/detectron2/dash_train_{idx}.png",
            vis.get_image()[:, :, ::-1],
        )

    # Train.
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATASETS.TRAIN = ("dash_train",)
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
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(
        SHAPE2INT
    )  # only has one class (ballon)
    cfg.OUTPUT_DIR = "ns_vqa_dart/scene_parse/detectron2/output"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # Validate.
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    trainer.resume_or_load(resume=True)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        0.7  # set the testing threshold for this model
    )
    cfg.DATASETS.TEST = ("dash_val",)
    predictor = DefaultPredictor(cfg)

    # Visualize val results.
    dataset_dicts = get_dash_dicts(split="val")
    dash_metadata = MetadataCatalog.get("dash_val")
    for idx, d in enumerate(random.sample(dataset_dicts, 1)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(
            im[:, :, ::-1],
            metadata=dash_metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(
            f"ns_vqa_dart/scene_parse/detectron2/dash_val_{idx}.png",
            v.get_image()[:, :, ::-1],
        )

    # Evaluate performance.
    dataset_dicts = DatasetCatalog.get("dash_val")
    for image_id, image_dict in enumerate(dataset_dicts):
        for annotation in image_dict["annotations"]:
            if "segmentation" in annotation:
                segmentation = annotation["segmentation"]
                if isinstance(segmentation, list):
                    print("segmentation")
                elif isinstance(segmentation, dict):
                    area = pycocotools.mask.area(segmentation)
                    print("rle")
                    print(f"area: {area}")
                    # with open("/home/michelle/test.json", "w") as f:
                    #     json.dump({
                    #         "segmentation": segmentation
                    #     }, f)
    metadata = MetadataCatalog.get("dash_val")
    evaluator = COCOEvaluator("dash_val", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "dash_val")
    res = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(res)


if __name__ == "__main__":
    main()
