"""Converts the DASH dataset into a format for detectron2 training, and trains
a detectron2 model on the dataset.
"""
import os
import cv2
import json
import pickle
import pprint
import random
import imageio
import argparse
import numpy as np
from typing import *
from tqdm import tqdm
from datetime import datetime

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


class DASHTrainer:
    def __init__(self, seed: int, root_dir: str):
        self.root_dir = root_dir
        random.seed(seed)

        self.cfg = self.get_cfg()
        self.n_visuals = 15

    def get_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        # Data configurations.
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.DATASETS.TRAIN = ("dash_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2

        # Model configurations.
        # Let training initialize from model zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        # Training configurations.
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 100000
        cfg.SOLVER.CHECKPOINT_PERIOD = 5000
        return cfg

    def train(self):
        # Output configurations.
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.cfg.OUTPUT_DIR = os.path.join(self.root_dir, time_str)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def eval(self, split: str, model_name: str):
        """Runs evaluation.

        Args:
            split: The split to evaluate.
            model_name: The name of the model to evaluate
        """
        # Model configurations.
        assert model_name
        self.cfg.OUTPUT_DIR = os.path.join(self.root_dir, model_name)
        with open(
            os.path.join(self.cfg.OUTPUT_DIR, "last_checkpoint"), "r"
        ) as f:
            checkpoint_fname = f.readlines()[0]
        self.cfg.MODEL.WEIGHTS = os.path.join(
            self.cfg.OUTPUT_DIR, checkpoint_fname
        )
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=True)

        # Dataset configurations.
        dataset_name = f"dash_{split}"
        self.cfg.DATASETS.TEST = (dataset_name,)

        # set the testing threshold for this model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        self.visualize(split=split)

        # Compute metrics.
        evaluator = COCOEvaluator(
            dataset_name=dataset_name,
            cfg=self.cfg,
            distributed=False,
            output_dir=self.cfg.OUTPUT_DIR,
        )
        val_loader = build_detection_test_loader(self.cfg, dataset_name)
        res = inference_on_dataset(trainer.model, val_loader, evaluator)
        return res

    def visualize(self, split: str):
        # Dataset.
        dataset_name = f"dash_{split}"
        dataset_dicts = get_dash_dicts(split=split)
        dash_metadata = MetadataCatalog.get(dataset_name)

        vis_dir = os.path.join(self.cfg.OUTPUT_DIR, f"images/{dataset_name}")
        os.makedirs(vis_dir, exist_ok=True)

        predictor = DefaultPredictor(self.cfg)
        for idx, d in enumerate(
            random.sample(
                dataset_dicts, min(self.n_visuals, len(dataset_dicts))
            )
        ):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(
                im[:, :, ::-1],
                metadata=dash_metadata,
                scale=1.0,
                # instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(
                os.path.join(vis_dir, f"{idx:02}.png"),
                v.get_image()[:, :, ::-1],
            )


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
        end_idx = 16000
    elif split == "val":
        start_idx = 16000
        end_idx = 20000
    print(f"Loading the dataset for split {split}...")

    for dataset in [
        "planning_v003_20K",
        "placing_v003_2K_20K",
        "stacking_v003_2K_20K",
    ]:
        for idx in tqdm(range(start_idx, end_idx)):
            record = {}

            # Construct the full path of the image.
            rgb_path = f"/media/sdc3/mguo/data/datasets/{dataset}/unity_output/images/first/rgb/{idx:06}_0.png"
            seg_path = f"/media/sdc3/mguo/data/datasets/{dataset}/unity_output/images/first/seg/{idx:06}_0.png"

            # Retrieve the height and width of the image.
            seg_img = imageio.imread(seg_path)
            height, width = imageio.imread(rgb_path).shape[:2]

            record["file_name"] = rgb_path
            record["image_id"] = f"{dataset}_{idx:06}"
            record["height"] = height
            record["width"] = width

            # Compute the segmentations.
            seg_map, oids = seg_img_to_map(seg_img=seg_img)

            objs = []
            for oid in oids:
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
                    "category_id": 0,
                    "iscrowd": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


def main(args: argparse.Namespace):
    # Register the datasets.
    for d in ["train", "val"]:
        # Associate a dataset named "dash_{split}" with the function that
        # returns the data for the split.
        DatasetCatalog.register(
            "dash_" + d, lambda d=d: get_dash_dicts(split=d)
        )
        MetadataCatalog.get("dash_" + d).set(thing_classes=["object"])

    trainer = DASHTrainer(seed=args.seed, root_dir=args.root_dir)
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        # train_res = trainer.eval(split="train", model_name=args.model_name)
        val_res = trainer.eval(split="val", model_name=args.model_name)

        # print("Train Results:")
        # pprint.pprint(train_res)
        print("Validation Results:")
        pprint.pprint(val_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        choices=["train", "eval"],
        help="Whether to train or run evaluation.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/sdc3/mguo/outputs/detectron",
        help="The root directory containing models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="2020_04_27_20_12_14",
        help="The name of the model to evaluate.",
    )
    args = parser.parse_args()
    main(args=args)
