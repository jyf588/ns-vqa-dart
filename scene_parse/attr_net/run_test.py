import os
import json
import numpy as np
from tqdm import tqdm

from options import get_options
from datasets import get_dataloader
from model import get_model

import bullet.util


def main():
    opt = get_options("test")
    test_loader = get_dataloader(opt, "test")
    model = get_model(opt)

    count = 0
    preds = []
    for data, _, img_ids, oids in tqdm(test_loader):
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
    print("%d / %d objects processed" % (count, len(test_loader.dataset)))
    print("| saving annotation file to %s" % opt.output_path)
    bullet.util.save_json(path=opt.output_path, data=preds)


if __name__ == "__main__":
    main()
