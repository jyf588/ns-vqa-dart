"""Generates and saves input data for an existing dataset."""
from tqdm import tqdm

from bullet.dash_dataset import DashDataset

DATASET_DIR = "/home/michelle/datasets/ego_v008"


def main():
    dataset = DashDataset(dataset_dir=DATASET_DIR)
    for eid in tqdm(range(17200, 22000)):
        objects, camera, rgb, mask = dataset.load_example_for_eid(eid=eid)
        dataset.save_input_data(objects=objects, rgb=rgb, mask=mask)


if __name__ == "__main__":
    main()
