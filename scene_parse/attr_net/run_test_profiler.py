import json
import numpy as np
import os
import sys
import time
import torch
import torchvision.transforms as transforms
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

    # print(f"Warning: Predictions are not currently being saved.")

    n_batches = 100

    count = 0
    preds = []
    start = time.time()
    for batch_iter, data in enumerate(tqdm(test_loader)):
        # for batch_iter in tqdm(range(n_batches)):
        #     data = np.zeros((7, 6, 480, 480), dtype=np.float32)
        #     # data = (data - 0.5) / 0.225

        #     for i in range(len(data)):
        #         data[i] = transforms.Compose(
        #             [
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(mean=[0.5] * 6, std=[0.225] * 6),
        #             ]
        #         )(np.zeros((480, 480, 6), dtype=np.float32))
        #     data = torch.Tensor(data)

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
        if batch_iter == n_batches - 1:
            break
    total_time = time.time() - start
    print(f"total time: {total_time:.2f} seconds.")
    print(f"Average time/iter: {(total_time/n_batches)*1000:.2f} ms.")

    print(set_input_prof)
    print(forward_prof)
    print(get_pred_prof)

    # print("%d / %d objects processed" % (count, len(test_loader.dataset)))
    # print("| saving annotation file to %s" % opt.output_path)
    # bullet.util.save_json(path=opt.output_path, data=preds)


if __name__ == "__main__":
    main()
