"""Visualizes images and labels for a particular dataset."""
import imageio
from matplotlib.pyplot import cm
import numpy as np
import os
from tqdm import tqdm

from bullet.dash_dataset import DashDataset


def main(dataset_name: str):
    dataset_dir = f"/home/michelle/datasets/{dataset_name}"
    output_dir = f"/home/michelle/analysis/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Create an instance of a DashDataset.
    dataset = DashDataset(dataset_dir=dataset_dir)

    # For each example, load the rgb image and mask.
    eids = dataset.load_example_ids()
    for eid in tqdm(eids):
        rgb, mask = dataset.load_example(eid=eid)
        mask_img = convert_mask_to_img(mask=mask)
        visual = np.hstack([rgb, mask_img])
        path = os.path.join(output_dir, f"{eid:05}.png")
        imageio.imwrite(path, visual)


def convert_mask_to_img(mask: np.ndarray):
    H, W = mask.shape
    mask_img = np.zeros((H, W, 3)).astype(np.uint8)
    oids = np.unique(mask)
    color = iter(cm.rainbow(np.linspace(0, 1, len(oids))))
    for oid in oids:
        mask_img[mask == oid] = next(color)[:3] * 255
    return mask_img


if __name__ == "__main__":
    dataset_name = "ego_001"
    main(dataset_name=dataset_name)
