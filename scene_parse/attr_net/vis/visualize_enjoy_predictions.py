import os
from tqdm import tqdm
import time

import cv2
import imageio
import json
import numpy as np
import pybullet as p

import vis.bullet
import vis.util


src_dir = '/home/michelle/outputs/enjoy_large_box'
dst_dir = '/home/michelle/outputs/enjoy_large_box_processed'
os.makedirs(dst_dir, exist_ok=True)

id2name = {
    '1': 'large_box',
    '2': 'large_cylinder'
}

id2color = {
    '1': 'red',
    '2': 'blue'
}

def main():
    for img_fname in tqdm(os.listdir(src_dir)):
        # If it's an image
        if img_fname.endswith('.png') and img_fname.startswith('s'):
            image_path = os.path.join(src_dir, img_fname)

            # Find the corresponding labels file.
            example_id = img_fname[1:-4]
            json_path = os.path.join(src_dir, f'{example_id}.json')

            json_dict = json.load(open(json_path, 'r'))

            img = imageio.imread(image_path)

            line_idx = 0
            render_gt_dict = {}
            render_pred_dict = {}
            for idx, (obj_id, (gt_position, gt_rotation)) in enumerate(json_dict['gt'].items()):
                gt_up_vector = np.array(gt_rotation).reshape((3, 3))[:, -1]
                pred_position = json_dict['preds'][obj_id]['position']
                pred_up_vector = json_dict['preds'][obj_id]['up_vector']
                pred_rotation = np.eye(3)
                pred_rotation[:, -1] = pred_up_vector
                pred_rotation = pred_rotation.reshape((9,))

                render_gt_dict[obj_id] = gt_position, gt_rotation
                render_pred_dict[obj_id] = pred_position, pred_rotation

                lines = [
                    f"{id2name[obj_id]}:",
                    f"    GT Pos: {vis.util.vec_to_str(gt_position)}",
                    f"    Pred Pos: {vis.util.vec_to_str(pred_position)}",
                    f"    GT Up: {vis.util.vec_to_str(gt_up_vector)}",
                    f"    Pred Up: {vis.util.vec_to_str(pred_up_vector)}",
                ]
                for line in lines:
                    img = cv2.putText(img,
                        line,
                        (25, 25 + line_idx * 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.8,
                        (255, 0, 0),
                    )
                    line_idx += 1

            # Render GT
            for obj_id, (pos, rot) in render_gt_dict.items():
                vis.bullet.render_bullet_object(id2name[obj_id], pos, rot, color=id2color[obj_id])
            # Render pred
            vis.bullet.render_bullet_object('tabletop', [0.25, 0.2, 0.0], color='grey')
            gt_rendered_img = vis.bullet.render_image()
            p.resetSimulation()

            for obj_id, (pos, rot) in render_pred_dict.items():
                vis.bullet.render_bullet_object(id2name[obj_id], pos, rot, color=id2color[obj_id])
            # Render pred
            vis.bullet.render_bullet_object('tabletop', [0.25, 0.2, 0.0], color='grey')
            pred_rendered_img = vis.bullet.render_image()
            p.resetSimulation()

            # Model input arrays
            # np_data = np.load(os.path.join(src_dir, f'{example_id}.npy'))
            # top_npy = np_data[0]
            # btm_npy = np_data[1]
            # top_img = top_npy[3:6, 80:400, :]
            # # top_seg = top_npy[3:]
            # btm_img = btm_npy[3:6, 80:400, :]
            # # btm_seg = btm_npy[3:]

            # Save the final image.
            img = np.hstack([img, gt_rendered_img, pred_rendered_img])
            outpath = os.path.join(dst_dir, img_fname)
            imageio.imwrite(outpath, img)


if __name__ == '__main__':
    p.connect(p.DIRECT)
    vis.bullet.init_camera()

    main()

    # for i in range (10000):
    #     p.stepSimulation()
    #     time.sleep(1)
    p.disconnect()
