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
    for X, Y in tqdm(test_loader):
        model.set_input(X)
        model.forward()
        outputs = model.get_pred()
        for i in range(outputs.shape[0]):
            pred = outputs[i].tolist()
            y = Y[i].tolist()
            preds.append(pred)
            labels.append(y)
        count += outputs.shape[0]

    metrics = Metrics()
    for y_hat, y in zip(preds, labels):
        gt_dict = dash_object.y_vec_to_dict(y=y, coordinate_frame="world",)
        pred_dict = dash_object.y_vec_to_dict(y=y_hat, coordinate_frame="world",)
        metrics.add_example(gt_dict, pred_dict)
    metrics.print()


if __name__ == "__main__":
    main()
