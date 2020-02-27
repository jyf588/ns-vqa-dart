import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm

from options import get_options
from datasets import get_dataloader
from model import get_model

import bullet.util

sys.path.append("/home/michelle/workspace/ns-vqa-dart")
from bullet.profiler import Profiler
import bullet.metrics


def main():
    opt = get_options("test")
    test_loader = get_dataloader(opt, "test")
    model = get_model(opt)
    model.eval_mode()

    # print(f"Warning: Predictions are not currently being saved.")

    count = 0
    preds = []
    start = time.time()
    for batch_iter, (data, img_ids, oids) in enumerate(tqdm(test_loader)):
        model.set_input(data)
        model.forward()
        pred = model.get_pred()

        for i in range(pred.shape[0]):
            img_id = img_ids[i].item()
            oid = oids[i].item()
            pred_vec = pred[i].tolist()
            preds.append({"img_id": img_id, "oid": oid, "pred": pred_vec})
        assert oids.size(0) == pred.shape[0]
        count += oids.size(0)
    total_time = time.time() - start

    print("%d / %d objects processed" % (count, len(test_loader.dataset)))
    print("| saving annotation file to %s" % opt.output_path)
    bullet.util.save_json(path=opt.output_path, data=preds)

    print("Computing metrics:")
    bullet.metrics.compute_metrics(
        dataset_dir=opt.dataset_dir,
        pred_dicts=preds,
        coordinate_frame=opt.coordinate_frame,
    )


if __name__ == "__main__":
    main()

