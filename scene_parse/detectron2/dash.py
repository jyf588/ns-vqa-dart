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

import torch
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

import exp.loader
from ns_vqa_dart.bullet import util
from ns_vqa_dart.bullet.seg import seg_img_to_map


class DASHSegModule:
    def __init__(
        self,
        mode: str,
        dataset_name: str,
        train_root_dir: Optional[str] = None,
        resume_from_dir=None,
        checkpoint_path: Optional[str] = None,
        vis_dir: Optional[str] = None,
        n_visuals: Optional[int] = 30,
        seed=None,
    ):
        """
        Args:
            mode: Whether to use this module for training or inference.
            train_root_dir: The directory to save training models to.
            seed: The random seed.
            vis_dir: The directory to save visuals to.
            n_visuals: Number of examples to generate visuals for.
        """
        self.mode = mode
        self.dataset_name = dataset_name
        self.train_root_dir = train_root_dir
        self.resume_from_dir = resume_from_dir
        self.checkpoint_path = checkpoint_path
        self.vis_dir = vis_dir
        self.n_visuals = n_visuals

        if seed is not None:
            self.seed_everything(seed, mode)

        self.cfg = self.get_cfg()

        if mode == "train":
            self.set_train_cfg()
        elif mode in ["eval", "eval_single"]:
            self.set_eval_cfg()
        else:
            raise ValueError(f"Invalid mode: {mode}.")

    def seed_everything(self, seed, mode):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Only set this during eval, since it could slow training.
        if mode == "eval":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def get_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )

        # Data configurations.
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.DATALOADER.NUM_WORKERS = 2

        # Model configurations.
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        return cfg

    def set_train_cfg(self):
        torch.backends.cudnn.benchmark = True

        self.cfg.DATASETS.TRAIN = (self.dataset_name,)

        # Let training initialize from model zoo
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        self.cfg.SOLVER.IMS_PER_BATCH = 2 * 2
        self.cfg.SOLVER.BASE_LR = 0.00025 * 2
        self.cfg.SOLVER.MAX_ITER = 600000
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500

        self.cfg.OUTPUT_DIR = os.path.join(self.train_root_dir, util.get_time_dirname())
        os.makedirs(self.cfg.OUTPUT_DIR)

    def set_eval_cfg(self):
        assert self.checkpoint_path is not None
        self.cfg.MODEL.WEIGHTS = self.checkpoint_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

        if self.mode == "eval_single":
            DatasetCatalog.register(
                "eval_single", lambda: get_dash_dicts(exp_name="eval_single")
            )
            MetadataCatalog.get("eval_single").set(thing_classes=["object"])
            self.cfg.DATASETS.TEST = ("eval_single",)
        else:
            self.cfg.DATASETS.TEST = (self.dataset_name,)
            self.cfg.OUTPUT_DIR = os.path.join(
                os.path.dirname(self.checkpoint_path),
                "eval",
                self.exp_name,
                util.get_time_dirname(),
            )
            os.makedirs(self.cfg.OUTPUT_DIR)

    def train(self):
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def eval_example(self, bgr: np.ndarray, vis_id: Optional[int] = None):
        """Evaluates a single example.

        Args:
            checkpoint_path: The checkpoint path.
        
        Returns:
            masks: A numpy array of shape (N, H, W) of instance masks.
        """
        predictor = DefaultPredictor(self.cfg)
        outputs = predictor(bgr)

        if vis_id is not None and self.vis_dir is not None:
            self.visualize_predictions(bgr=bgr, outputs=outputs, vis_id=vis_id)

        # This produces a numpy array of shape (N, H, W) containing binary
        # masks.
        masks = self.get_output_masks(outputs=outputs)
        return masks

    def get_output_masks(self, outputs):
        masks = outputs["instances"].to("cpu")._fields["pred_masks"].numpy()
        return masks

    def eval(
        self,
        compute_metrics: Optional[bool] = False,
        visualize: Optional[bool] = False,
        save_segs: Optional[bool] = False,
    ):
        """Runs evaluation.

        Args:
            split: The split to evaluate.
            checkpoint_path: The checkpoint path.
        """
        dataset_dicts = get_dash_dicts(exp_name=self.exp_name)
        n = len(dataset_dicts)

        predictor = DefaultPredictor(self.cfg)

        idxs_to_eval, visualize_idxs, save_seg_idxs = [], [], []
        if visualize:
            visualize_idxs = random.sample(range(n), min(self.n_visuals, n))
            idxs_to_eval += visualize_idxs
        if save_segs:
            save_seg_idxs = list(range(len(dataset_dicts)))
        idxs_to_eval = set(visualize_idxs) | set(save_seg_idxs)

        # Get scene_id to timestep mapping.
        set2scene_id2timesteps = {}
        for idx in idxs_to_eval:
            d = dataset_dicts[idx]
            set_name, scene_id, timestep = d["image_id"].split("_")
            timestep = int(timestep)
            if set_name not in set2scene_id2timesteps:
                set2scene_id2timesteps[set_name] = {}
            if scene_id not in set2scene_id2timesteps[set_name]:
                set2scene_id2timesteps[set_name][scene_id] = []
            set2scene_id2timesteps[set_name][scene_id].append((idx, timestep))

        print("Evaluating dataset...")
        for set_name in set2scene_id2timesteps.keys():
            for scene_id in set2scene_id2timesteps[set_name].keys():
                print(
                    f"Evaluating {self.exp_name}\tset: {set_name}\tscene ID: {scene_id}"
                )
                scene_loader = exp.loader.SceneLoader(
                    exp_name=self.exp_name, set_name=set_name, scene_id=scene_id
                )
                os.makedirs(scene_loader.detectron_masks_dir)

                for idx, ts in tqdm(set2scene_id2timesteps[set_name][scene_id]):
                    d = dataset_dicts[idx]

                    # Predictor always takes in a BGR image.
                    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py#L162
                    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py#L212
                    bgr = cv2.imread(d["file_name"])
                    outputs = predictor(bgr)

                    if idx in save_seg_idxs:
                        masks = self.get_output_masks(outputs=outputs)
                        if idx == 0:
                            print(f"Saving detectron masks. Showing first example...")
                            m_img = np.hstack([m.astype(np.uint8) * 255 for m in masks])
                            vis = self.visualize_predictions(
                                bgr=bgr,
                                outputs=outputs,
                                vis_id=d["image_id"],
                                save=False,
                            )
                            cv2.imshow("masks", m_img)
                            cv2.imshow("BGR", np.hstack([bgr, vis]))
                            k = cv2.waitKey(0)
                            cv2.destroyWindow("masks")
                            cv2.destroyWindow("BGR")
                        scene_loader.save_detectron_masks(timestep=ts, masks=masks)

                    if idx in visualize_idxs:
                        self.visualize_predictions(
                            bgr=bgr, outputs=outputs, vis_id=d["image_id"]
                        )

        # Compute metrics.
        if compute_metrics:
            res = self.compute_metrics()
            return res

    def visualize_predictions(
        self, bgr: np.ndarray, outputs: Dict, vis_id: str, remove_scores=True, save=True
    ):
        """

        Args:
            img (np.ndarray): An image of shape (H, W, C), in BGR order.
            outputs: The outputs of the model.
        """
        metadata = MetadataCatalog.get(self.exp_name)

        # Remove scores from the visualization.
        if remove_scores:
            fields = outputs["instances"]._fields
            fields_wo_scores = {}
            for key in ["pred_boxes", "pred_classes", "pred_masks"]:
                fields_wo_scores[key] = fields[key]
            outputs["instances"]._fields = fields_wo_scores

        # Generate the visualization.
        v = Visualizer(
            bgr, metadata=metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION,
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result = v.get_image()

        if save:
            if self.vis_dir is None:
                self.vis_dir = os.path.join(self.cfg.OUTPUT_DIR, "images")
            os.makedirs(self.vis_dir, exist_ok=True)

            # Visualizer produces BGR.
            cv2.imwrite(os.path.join(self.vis_dir, f"{vis_id}.png"), result)
        return result

    def compute_metrics(self):
        evaluator = COCOEvaluator(
            dataset_name=self.exp_name,
            cfg=self.cfg,
            distributed=False,
            output_dir=self.cfg.OUTPUT_DIR,
        )
        val_loader = build_detection_test_loader(self.cfg, self.exp_name)

        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=True)
        res = inference_on_dataset(trainer.model, val_loader, evaluator)
        return res


def get_dash_dicts(set2dir: Dict, split: str, split_frac=0.8) -> List[Dict]:
    """Prepares the dataset dictionaries for the DASH dataset for detectron2
    training.

    Args:
        set2dir: A mapping from set name to image and segmentation image directories.
        split: Whether to create train or val split. Default will sort the files in each
            `data_dir`, and split first 80% to train and last 20% to val, for each dir.

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

    # Collect sorted examples from the various sets. We sort because that way splitting
    # is deterministic.
    example_sets = []
    for set_name, set_dirs in set2dir.items():
        set_examples = collect_sorted_examples_for_set(set_name, set_dirs)
        example_sets.append(set_examples)

    # Only include examples for the current split.
    examples = []
    for s in example_sets:
        split_examples = util.compute_split(
            split=split, examples=s, split_frac=split_frac
        )
        examples += split_examples

    # Convert examples into records.
    for set_name, sid, img_path, seg_img_path in tqdm(examples):
        image_id = f"{set_name}_{sid}"
        seg_img = imageio.imread(seg_img_path)
        masks, _ = seg_img_to_map(seg_img)

        # Create the record.
        record = create_record_for_image(
            image_id=image_id, img_path=img_path, masks=masks
        )
        dataset_dicts.append(record)
    return dataset_dicts


def collect_sorted_examples_for_set(set_name, set_dirs):
    img_dir = set_dirs["img"]
    seg_dir = set_dirs["seg"]

    # Verify that both directories have the same set of filenames.
    img_fnames = set(os.listdir(img_dir))
    seg_fnames = set(os.listdir(seg_dir))
    assert img_fnames == seg_fnames

    examples = []
    for fname in sorted(os.listdir(img_dir)):
        sid_with_zero, _ = fname.split(".")
        sid, _ = sid_with_zero.split("_")
        img_path = os.path.join(img_dir, fname)
        seg_path = os.path.join(seg_dir, fname)
        examples.append((set_name, sid, img_path, seg_path))
    return examples


def create_record_for_image(
    image_id: str, img_path: str, masks: np.ndarray, height=320, width=480
):
    record = {}
    record["file_name"] = img_path
    record["image_id"] = image_id
    record["height"] = height
    record["width"] = width

    objs = []
    for m in masks:
        rle = pycocotools.mask.encode(np.asarray(m, order="F"))

        # Convert `counts` to ascii, otherwise json dump complains about not being able
        # to serialize bytes.
        # https://github.com/facebookresearch/detectron2/issues/200#issuecomment-614407341
        # rle["counts"] = rle["counts"].decode("ASCII")
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
    return record


def get_latest_checkpoint(root_dir: str, exp_name: str, model_name: str):
    model_dir = os.path.join(root_dir, exp_name, model_name)
    with open(os.path.join(model_dir, "last_checkpoint"), "r",) as f:
        checkpoint_fname = f.readlines()[0]
    checkpoint_path = os.path.join(model_dir, checkpoint_fname)
    return checkpoint_path


def register_dataset(dataset_name, set2dir, split):
    DatasetCatalog.register(dataset_name, lambda: get_dash_dicts(set2dir, split))
    MetadataCatalog.get(dataset_name).set(thing_classes=["object"])


def main(args: argparse.Namespace):
    # Register the train and validation datasets.
    set_names = ["planning_v003_20K", "placing_v003_2K_20K", "stacking_v003_2K_20K"]
    set2dir = {}
    for n in set_names:
        images_dir = os.path.join("/home/mguo/data", n, "unity_output/images")
        img_dir = os.path.join(images_dir, "rgb")
        seg_dir = os.path.join(images_dir, "seg")
        set2dir[n] = {"img": img_dir, "seg": seg_dir}

    for split in ["train", "val"]:
        dataset_name = f"dash_{split}"
        register_dataset(dataset_name=dataset_name, set2dir=set2dir, split="train")

    if args.mode == "train":
        module = DASHSegModule(
            mode=args.mode, dataset_name="dash_train", train_root_dir=args.root_dir,
        )
        module.train()
    # elif args.mode == "eval":
    #     # Evaluate the latest checkpoint under the provided model name.
    #     checkpoint_path = get_latest_checkpoint(
    #         root_dir=args.root_dir, exp_name=args.exp_name, model_name=args.model_name
    #     )

    #     module = DASHSegModule(
    #         mode=args.mode,
    #         dataset_name="dash_val",
    #         checkpoint_path=checkpoint_path,
    #         seed=args.seed,  # Set seed since we want deterministic results.
    #     )

    #     res = module.eval(
    #         visualize=True, compute_metrics=True, save_segs=args.save_segs
    #     )
    #     print("Results:")
    #     pprint.pprint(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval"],
        help="Whether to train or run evaluation.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/mguo/outputs/detectron",
        help="The root directory containing models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="2020_05_11_11_56_20",
        help="The name of the model to evaluate.",
    )
    parser.add_argument(
        "--save_segs",
        action="store_true",
        help="Whether to save segmentation masks during evaluation.",
    )
    args = parser.parse_args()
    main(args=args)
