"""Updates labels for a dataset. Specifically converts the z positions from
H/2 to 0.
"""
from tqdm import tqdm

from bullet.dash_dataset import DashDataset


def main():
    dataset_dir = "/media/michelle/68B62784B62751BC/datasets/ego_v009"
    dataset = DashDataset(dataset_dir=dataset_dir)
    eids = dataset.load_example_ids()
    for eid in tqdm(eids):
        objects, camera = dataset.load_objects_and_camera_for_eid(eid=eid)
        for o in objects:
            o.position[2] = 0.0
        dataset.save_labels(objects=objects, camera=camera, eid=eid)


if __name__ == "__main__":
    main()
