import os
import json
import numpy as np

from options import get_options
from datasets import get_dataloader
from model import get_model

import bullet.util


def get_attrs_clevr_dart(opt, feat_vec):
    attr2choices = {
        "shape": ["box", "cylinder", "sphere"],
        "size": ["small", "large"],
        "color": ["red", "yellow", "green", "blue"],
    }

    start_idx = 0
    obj = {}
    if opt.pred_attr:
        for attr in ["color", "shape", "size"]:
            end_idx = start_idx + len(attr2choices[attr])
            pred_idx = np.argmax(feat_vec[start_idx:end_idx])
            obj[attr] = attr2choices[attr][pred_idx]
            start_idx = end_idx
    if opt.pred_position:
        end_idx = start_idx + 3
        obj["position"] = feat_vec[start_idx:end_idx].tolist()
        start_idx = end_idx
    if opt.pred_up_vector:
        end_idx = start_idx + 3
        obj["up_vector"] = feat_vec[start_idx:end_idx].tolist()
        start_idx = end_idx
    if opt.pred_height:
        obj["height"] = float(feat_vec[start_idx])

    return obj


def main():
    opt = get_options("test")
    test_loader = get_dataloader(opt, "test")
    model = get_model(opt)

    # scenes = [
    #     {
    #         "image_index": i,
    #         "image_filename": "s%05d.png" % (i + opt.split_id),
    #         "objects": [],
    #     }
    #     for i in range(2000)
    # ]  # TODO

    count = 0
    scenes = {}
    for data, _, img_ids, oids in test_loader:
        model.set_input(data)
        model.forward()
        pred = model.get_pred()
        for i in range(pred.shape[0]):
            img_id = img_ids[i].item()
            oid = oids[i].item()
            pred_vec = list(pred[i])
            print(type(pred_vec))
            if img_id not in scenes:
                scenes[img_id] = []
            scenes[img_id].append({"oid": oid, "pred": pred_vec})
        assert oids.size(0) == pred.shape[0]
        count += oids.size(0)
        print("%d / %d objects processed" % (count, len(test_loader.dataset)))
    print(scenes)
    print("| saving annotation file to %s" % opt.output_path)
    bullet.util.save_json(path=opt.output_path, data=scenes)


if __name__ == "__main__":
    main()
