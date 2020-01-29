import os
import json
import argparse
import numpy as np
import pycocotools.mask as mask_util

import torch

parser = argparse.ArgumentParser()
# parser.add_argument('--proposal_path', required=True, type=str)
parser.add_argument('--gt_scene_path', default=None, type=str)
parser.add_argument('--output_path', required=True, type=str)
# parser.add_argument('--align_iou_thresh', default=0.7, type=float)
# parser.add_argument('--score_thresh', default=0.9, type=float)
# parser.add_argument('--suppression', default=0, type=int)
# parser.add_argument('--suppression_iou_thresh', default=0.5, type=float)
# parser.add_argument('--suppression_iomin_thresh', default=0.5, type=float)


def get_feat_vec_clevr_dart(obj):
    attr_to_idx = {
        'box': 0,
        'cylinder': 1,
        'sphere': 2,
        'small': 3,
        'large': 4,
        'red': 5,
        'yellow': 6,
        'green': 7,
        'blue': 8
    }
    feat_vec = np.zeros(9+3)
    for attr in ['color', 'shape', 'size']:
        feat_vec[attr_to_idx[obj[attr]]] = 1
    feat_vec[9:12] = obj['position']

    # feat_vec[12:15] = obj['rotation_x'] - feat_vec[9:12]
    # feat_vec[15:18] = obj['rotation_y'] - feat_vec[9:12]
    # feat_vec[18:21] = obj['rotation_z'] - feat_vec[9:12]

    return list(feat_vec)


def load_clevr_dart_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    scenes = []
    for s in scenes_dict:
        objs = []
        for i, o in enumerate(s['objects']):
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)    # TODO: deprecated?
            if '3d_coords' in o:
                item['position'] = o['3d_coords']   # dart camera is not tilted
            else:
                item['position'] = o['position']

            # item['rotation_x'] = o['rotation_x']
            # item['rotation_y'] = o['rotation_y']
            # item['rotation_z'] = o['rotation_z']

            item['color'] = o['color']
            item['shape'] = o['shape']
            item['size'] = o['size']
            item['mask'] = o['mask']
            objs.append(item)
        scenes.append({
            'objects': objs,
        })
    return scenes


def iou(m1, m2):
    intersect = m1 * m2
    union = 1 - (1 - m1) * (1 - m2)
    return intersect.sum() / union.sum()


def main(args):
    output_dir = os.path.dirname(args.output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    scenes = None
    if args.gt_scene_path is not None:
        scenes = load_clevr_dart_scenes(args.gt_scene_path)    # load and rotate xyz camera

    image_width = 480
    image_height = 320  # TODO: hard coded size

    A = 0
    img_anns = []
    for image_id, scene in enumerate(scenes):
        obj_anns = []
        for o in scene['objects']:
            if mask_util.area(o['mask']) > 0:
                vec = get_feat_vec_clevr_dart(o)
                obj_ann = {
                    'mask': o['mask'],
                    'image_idx': image_id,
                    'feature_vector': vec,
                }
                obj_anns.append(obj_ann)
                A = A + 1
            else:
                print(image_id, o)
        img_anns.append(obj_anns)
        print('| processing proposals %d th image' % image_id)

    print(A)


    # # with open(args.proposal_path, 'rb') as f:
    # #     proposals = pickle.load(f)
    # # segms = proposals['all_segms']
    # # boxes = proposals['all_boxes'] # all we need from box is the score
    #
    # # TODO: pack all these post-processings to one function
    # predictions = torch.load(args.proposal_path)
    # image_width = 480
    # image_height = 320  # TODO: hard coded size
    #
    # # The generated masks are of low resolution- 28x28 pixels.
    # # During training, the masks are scaled down to 28x28 to compute
    # # the loss, and during inferencing, the predicted masks are scaled up
    # # to the size of the ROI bounding box.
    # masker = Masker(threshold=0.5, padding=1)
    #
    # img_anns = []
    # # image_id and original_id are the same in CLEVR
    # # for each image, see coco_eval where they are different
    # for image_id, prediction in enumerate(predictions):
    #
    #     prediction = prediction.resize((image_width, image_height))
    #     # TODO: probably not necessary since we do not care bb
    #     prediction = prediction.convert("xywh")
    #
    #     scores = prediction.get_field("scores").tolist()
    #     labels = prediction.get_field("labels").tolist()    # labels in int
    #
    #     masks = prediction.get_field("mask")
    #     # Masker is necessary only if masks haven't been already resized.
    #     # Here, prediction should return masks un-resized yet
    #     if list(masks.shape[-2:]) != [image_height, image_width]:
    #         print("resizing mask back to ROI size...")
    #         masks = masker(masks.expand(1, -1, -1, -1, -1), prediction) # return a list of tensors
    #         # masks = masker([masks], [prediction]) # TODO: seems no diff
    #         masks = masks[0]    # single image, single prediction
    #     # now mask is binary and has correct size
    #
    #     # what is "mask" format in the original implementation here?
    #     # https://github.com/kexinyi/ns-vqa/blob/master/scene_parse/attr_net/datasets/clevr_object.py#L58
    #     # according to above, should be in rleObj format
    #     rles = [
    #         mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
    #         for mask in masks
    #     ]
    #     for rle in rles:
    #         rle["counts"] = rle["counts"].decode('ASCII')   # TODO: or utf-8?
    #
    #     # boxes = prediction.bbox.tolist() # we do not care bb
    #
    #     obj_anns = []
    #     for obj_id, rle in enumerate(rles):
    #         # get rid of <0.9 objects during training
    #         if scores[obj_id] > args.score_thresh:
    #             if scenes is None:  # no ground truth alignment
    #                 obj_ann = {
    #                     'mask': rle,
    #                     'image_idx': image_id,
    #                     'category_idx': labels[obj_id],
    #                     'feature_vector': None,
    #                     'score': scores[obj_id],
    #                 }
    #                 obj_anns.append(obj_ann)
    #             else:
    #                 mask = masks[obj_id][0].numpy()     # odd..
    #                 # find the (first) gt_object that overlap significantly with inferred obj
    #                 # if not found, ignore predicted object (since we cannot annotate xyz for it)
    #                 for o in scenes[image_id]['objects']:
    #                     mask_gt = mask_util.decode(o['mask'])
    #                     # print(mask_gt.shape)
    #                     if iou(mask, mask_gt) > args.align_iou_thresh:
    #                         # input("found")
    #                         vec = get_feat_vec_clevr_dart(o)
    #                         obj_ann = {
    #                             'mask': rle,
    #                             'image_idx': image_id,
    #                             'category_idx': labels[obj_id],
    #                             'feature_vector': vec,
    #                             'score': scores[obj_id],
    #                         }
    #                         obj_anns.append(obj_ann)
    #                         break
    #
    #     img_anns.append(obj_anns)
    #     print('| processing proposals %d th image' % image_id)
    #
    # # # TODO: this seems to be turned off
    # # if scenes is None and args.suppression:
    # #     # Apply suppression on test proposals
    # #     all_objs = []
    # #     for i, img_ann in enumerate(img_anns):
    # #         objs_sorted = sorted(img_ann, key=lambda k: k['score'], reverse=True)
    # #         objs_suppressed = []
    # #         for obj_ann in objs_sorted:
    # #             if obj_ann['score'] > args.score_thresh:
    # #                 duplicate = False
    # #                 for obj_exist in objs_suppressed:
    # #                     mo = mask_util.decode(obj_ann['mask'])
    # #                     me = mask_util.decode(obj_exist['mask'])
    # #                     if utils.iou(mo, me) > args.suppression_iou_thresh \
    # #                        or utils.iomin(mo, me) > args.suppression_iomin_thresh:
    # #                         duplicate = True
    # #                         break
    # #                 if not duplicate:
    # #                     objs_suppressed.append(obj_ann)
    # #         all_objs += objs_suppressed
    # #         print('| running suppression %d / %d images' % (i+1, nimgs))
    # # else:
    # #     all_objs = [obj_ann for img_ann in img_anns for obj_ann in img_ann]

    all_objs = [obj_ann for img_ann in img_anns for obj_ann in img_ann]
    obj_masks = [o['mask'] for o in all_objs]
    img_ids = [o['image_idx'] for o in all_objs]
    if scenes is not None:
        feat_vecs = [o['feature_vector'] for o in all_objs]
    else:
        feat_vecs = []
    output = {
        'object_masks': obj_masks,
        'image_idxs': img_ids,
        'feature_vectors': feat_vecs,
    }
    print('| saving object annotations to %s' % args.output_path)
    with open(args.output_path, 'w') as fout:
        json.dump(output, fout, indent=2, separators=(',', ': '))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
