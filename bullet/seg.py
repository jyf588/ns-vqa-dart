import numpy as np

RGB2ID = {
    (174, 199, 232): 0,  # blue
    (152, 223, 138): 1,  # green
    (255, 152, 150): 2,  # red
    (197, 176, 213): 3,  # purple
    (219, 219, 141): 4,  # lime green
    (196, 156, 148): 5,  # brown
    (255, 187, 120): 6,  # orange
    (247, 182, 210): 7,  # pink
}


def seg_img_to_map(seg_img):
    H, W, _ = seg_img.shape
    seg = np.full((H, W), -1, dtype=np.uint8)
    ids = []
    for rgb_value, oid in RGB2ID.items():
        idxs = np.where(
            np.logical_and(
                seg_img[:, :, 0] == rgb_value[0],
                np.logical_and(
                    seg_img[:, :, 1] == rgb_value[1],
                    seg_img[:, :, 2] == rgb_value[2],
                ),
            )
        )
        seg[idxs] = oid
        if len(idxs[0]) > 0:
            ids.append(oid)
    return seg, ids
