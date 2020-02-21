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


def main():
    opt = get_options("test")
    test_loader = get_dataloader(opt, "test")
    model = get_model(opt)

    set_input_prof = Profiler("set_input")
    forward_prof = Profiler("forward")
    get_pred_prof = Profiler("get_pred")

    print(f"Warning: Predictions are not currently being saved.")

    count = 0
    preds = []
    start = time.time()
    for batch_iter, (data, img_ids, oids) in enumerate(tqdm(test_loader)):
        set_input_prof.start()
        model.set_input(data)
        set_input_prof.end()

        forward_prof.start()
        model.forward()
        forward_prof.end()

        get_pred_prof.start()
        pred = model.get_pred()
        get_pred_prof.end()

        # for i in range(pred.shape[0]):
        #     img_id = img_ids[i].item()
        #     oid = oids[i].item()
        #     pred_vec = pred[i].tolist()
        #     preds.append({"img_id": img_id, "oid": oid, "pred": pred_vec})
        # assert oids.size(0) == pred.shape[0]
        # count += oids.size(0)
    total_time = time.time() - start
    print(f"total time: {total_time:.2f} seconds.")
    print(f"Average time/iter: {(total_time/len(test_loader))*1000:.2f} ms.")

    print(set_input_prof)
    print(forward_prof)
    print(get_pred_prof)

    # print("%d / %d objects processed" % (count, len(test_loader.dataset)))
    # print("| saving annotation file to %s" % opt.output_path)
    # bullet.util.save_json(path=opt.output_path, data=preds)


if __name__ == "__main__":
    main()
