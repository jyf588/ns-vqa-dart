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

from ns_vqa_dart.bullet import util
from ns_vqa_dart.bullet import metrics


def main():
    opt = get_options("test")
    test_loader = get_dataloader(opt, "test")
    model = get_model(opt)
    model.eval_mode()

    count = 0
    sid2info = {}
    start = time.time()
    for batch_iter, (data, labels, sids, oids, paths) in enumerate(
        tqdm(test_loader)
    ):
        model.set_input(data)
        model.forward()
        pred = model.get_pred()
        for i in range(pred.shape[0]):
            path = paths[i]
            sid = sids[i].item()
            oid = oids[i].item()
            if sid not in sid2info:
                sid2info[sid] = {}
            sid2info[sid][oid] = {
                "pred": pred[i].tolist(),
                "labels": labels[i].tolist(),
            }
        count += pred.shape[0]
    total_time = time.time() - start

    # pprint.pprint(sid2info)

    print("%d / %d objects processed" % (count, len(test_loader.dataset)))
    print("| saving annotation file to %s" % opt.output_path)
    util.save_json(path=opt.output_path, data=sid2info)

    print("Computing metrics:")
    metrics.compute_metrics(
        cam_dir=opt.cam_dir,
        sid2info=sid2info,
        camera_control=opt.camera_control,
        coordinate_frame=opt.coordinate_frame,
        plot_path=opt.plot_path,
    )


if __name__ == "__main__":
    main()
