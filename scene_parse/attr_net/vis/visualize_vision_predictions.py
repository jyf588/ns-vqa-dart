import os
from tqdm import tqdm

import imageio
import json
import numpy as np
import pybullet as p

import vis.bullet
import vis.util
import vis.inference


dataset = 'enjoy_large_box'
model = 'xyz_up'
src_dir = f'/home/michelle/datasets/{dataset}/images'
pred_json_path = f'/home/michelle/outputs/{dataset}/derendered.json'
dst_dir = f'/home/michelle/outputs/{dataset}/pred'
start_i = 0
end_i = 1
split_id = 0
run_inference = True
load_checkpoint_path = '/home/michelle/outputs/placing_v4_direct/xyz_up/checkpoint_best.pt'
pred_format = 'xyz_up'
pred_pos_format = 'xyz'
pred_rot_format = 'up'


def main():
    os.makedirs(dst_dir, exist_ok=True)

    pred_json = json.load(open(pred_json_path, 'r'))
    all_infer_objects = vis.inference.get_inferred_objects()
    all_gt = []
    for test_idx in range(start_i, end_i):
        idx = test_idx + split_id
        input_img_path = os.path.join(src_dir, f"s{idx:05}.png")
        mask_dir = os.path.join(src_dir, f"ms{idx:05}")
        anno_path = os.path.join(src_dir, f"anno{idx:05}.json")
        output_img_path = os.path.join(dst_dir, f"{idx:05}.png")
        
        img = imageio.imread(input_img_path)

        gt_objects = json.load(open(anno_path, 'r'))['objects']
        pred_objects = pred_json['scenes'][test_idx]['objects']
        infer_objects = all_infer_objects[test_idx]

        # Rename the keys.
        for obj_idx in range(len(gt_objects)):
            gt_objects[obj_idx]['position'] = gt_objects[obj_idx].pop('world_pos')
            gt_objects[obj_idx]['rotation'] = gt_objects[obj_idx].pop('world_rot')
        
        for obj_idx in range(len(pred_objects)):
            pred_objects[obj_idx]['shape'] = gt_objects[obj_idx]['shape']
            pred_objects[obj_idx]['size'] = gt_objects[obj_idx]['size']
            pred_objects[obj_idx]['color'] = gt_objects[obj_idx]['color']
            pred_objects[obj_idx]['rotation'] = vis.util.up_vector_to_rotation(pred_objects[obj_idx]['z_dir'])

        for obj_idx in range(len(infer_objects)):
            infer_objects[obj_idx]['shape'] = gt_objects[obj_idx]['shape']
            infer_objects[obj_idx]['size'] = gt_objects[obj_idx]['size']
            infer_objects[obj_idx]['color'] = gt_objects[obj_idx]['color']
            infer_objects[obj_idx]['rotation'] = vis.util.up_vector_to_rotation(infer_objects[obj_idx]['z_dir'])

        gt_rendered = vis.bullet.render_scene(gt_objects, table=True)
        pred_rendered = vis.bullet.render_scene(pred_objects, table=True)
        infer_rendered = vis.bullet.render_scene(infer_objects, table=True)
        img = np.hstack([img, gt_rendered, pred_rendered, infer_rendered])
        imageio.imwrite(output_img_path, img)
        p.resetSimulation()


if __name__ == '__main__':
    p.connect(p.DIRECT)
    vis.bullet.init_camera()
    main()
    p.disconnect()

# images = []
# masks = []
# img = imageio.imread(input_img_path)
# for mask_i in range(2):
#     mask = imageio.imread(os.path.join(mask_dir, f"ob{mask_i:02}.png"))
#     images.append(img)
#     masks.append(mask)
# infer_objects = vis.inference.eval_with_loader(images, masks, load_checkpoint_path, pred_format, pred_pos_format, pred_rot_format)
