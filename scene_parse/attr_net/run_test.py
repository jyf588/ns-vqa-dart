"""Runs forward prediction on a test set.

Outputs a json file with the following format: {
    <sid>: {
        <oid>: {
            "pred": <pred: List>,
            "labels": <labels: List>
        }
    }
}
"""

import json
import numpy as np
import os
import pprint
import sys
import time
from tqdm import tqdm

from options import get_options
from datasets import get_dataloader
from model import get_model

import exp.loader
from ns_vqa_dart.bullet import util, dash_object
from ns_vqa_dart.bullet.metrics import Metrics


def main():
    opt = get_options("test")
    test_loader = get_dataloader(opt, "test")
    model = get_model(opt)
    model.eval_mode()

    count = 0
    preds = []
    labels = []
    meta_list = []
    for X, Y, all_meta in tqdm(test_loader):
        model.set_input(X)
        model.forward()
        outputs = model.get_pred()
        for i in range(outputs.shape[0]):
            pred = outputs[i].tolist()
            y = Y[i].tolist()
            set_name = all_meta["set_name"][i]
            scene_id = all_meta["scene_id"][i]
            timestep = int(all_meta["timestep"][i])
            oid = int(all_meta["oid"][i])
            meta_list.append((set_name, scene_id, timestep, oid))
            preds.append(pred)
            labels.append(y)
        count += outputs.shape[0]

    local_metrics = Metrics()
    world_metrics = Metrics()
    for y_hat, y, meta in zip(preds, labels, meta_list):
        gt_dict = dash_object.y_vec_to_dict(y=y, coordinate_frame="world",)
        pred_dict = dash_object.y_vec_to_dict(y=y_hat, coordinate_frame="world",)
        local_metrics.add_example(gt_dict, pred_dict)

        set_name, scene_id, timestep, oid = meta
        scene_loader = exp.loader.SceneLoader(
            exp_name=opt.eval_set, set_name=set_name, scene_id=scene_id
        )
        odict = scene_loader.load_odict(timestep=timestep, oid=oid)
        cam_dict = scene_loader.load_cam(timestep=timestep)
        cam_position = cam_dict["position"]
        cam_orientation = cam_dict["orientation"]
        gt_dict = dash_object.y_vec_to_dict(
            y=y,
            coordinate_frame=opt.coordinate_frame,
            cam_position=cam_position,
            cam_orientation=cam_orientation,
        )
        pred_dict = dash_object.y_vec_to_dict(
            y=y_hat,
            coordinate_frame=opt.coordinate_frame,
            cam_position=cam_position,
            cam_orientation=cam_orientation,
        )
        print("****************")
        print(f"meta: {meta}")
        print("pred_dict")
        pprint.pprint(pred_dict)
        print("gt_dict")
        pprint.pprint(gt_dict)
        print("odict")
        pprint.pprint(odict)
        # world_metrics.add_example(gt_dict, pred_dict)
        world_metrics.add_example(odict, pred_dict)
    local_metrics.print()
    world_metrics.print()


if __name__ == "__main__":
    main()
