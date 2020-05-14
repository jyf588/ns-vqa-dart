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
import os
import sys
import time
import pprint
import numpy as np
from tqdm import tqdm

from options import get_options
from model import get_model
from datasets import get_dataloader
from ns_vqa_dart.bullet import util


def main():
    opt = get_options("test")
    test_loader = get_dataloader(opt, "test")
    model = get_model(opt)
    model.eval_mode()

    path = os.path.join(opt.run_dir, "preds.p")

    predictions = {}
    for X, Y, sids in tqdm(test_loader):
        model.set_input(X)
        model.forward()
        outputs = model.get_pred()
        sids = sids.tolist()
        for i in range(outputs.shape[0]):
            predictions[sids[i]] = {
                "pred": outputs[i].tolist(),
                "y": Y[i].tolist(),
            }
    util.save_pickle(path=path, data=predictions)

    # local_metrics = Metrics()
    # world_metrics = Metrics()
    # for y_hat, y, meta in zip(preds, labels, meta_list):
    #     gt_dict = dash_object.y_vec_to_dict(y=y, coordinate_frame="world",)
    #     pred_dict = dash_object.y_vec_to_dict(y=y_hat, coordinate_frame="world",)
    #     local_metrics.add_example(gt_dict, pred_dict)

    #     set_name, scene_id, timestep, oid = meta
    #     scene_loader = exp.loader.SceneLoader(
    #         exp_name=opt.eval_set, set_name=set_name, scene_id=scene_id
    #     )
    #     odict = scene_loader.load_odict(timestep=timestep, oid=oid)
    #     cam_dict = scene_loader.load_cam(timestep=timestep)
    #     cam_position = cam_dict["position"]
    #     cam_orientation = cam_dict["orientation"]
    #     gt_dict = dash_object.y_vec_to_dict(
    #         y=y,
    #         coordinate_frame=opt.coordinate_frame,
    #         cam_position=cam_position,
    #         cam_orientation=cam_orientation,
    #     )
    #     pred_dict = dash_object.y_vec_to_dict(
    #         y=y_hat,
    #         coordinate_frame=opt.coordinate_frame,
    #         cam_position=cam_position,
    #         cam_orientation=cam_orientation,
    #     )
    #     world_metrics.add_example(odict, pred_dict)
    # local_metrics.print()
    # world_metrics.print()


if __name__ == "__main__":
    main()
