import numpy as np
import os

import cv2
import imageio
import torch
import torchvision.transforms as transforms
import pycocotools.mask as mask_util
from torch.utils.data import DataLoader

from datasets import get_dataloader
from model import get_model
from run_test import get_attrs_clevr_dart


def get_inferred_objects():
    scenes = {}
    # opt = get_options('test')
    opt = get_inference_options()
    test_loader = get_dataloader(opt, 'test')
    model = get_model(opt)
    count = 0
    for data, _, idxs, cat_idxs in test_loader:
        # each data should be one obj and its scene
        model.set_input(data)
        model.forward()
        pred = model.get_pred()
        for i in range(pred.shape[0]):
            obj = get_attrs_clevr_dart(opt, pred[i])
            scene_id = int(idxs[i].numpy()) - opt.split_id
            if scene_id not in scenes:
                scenes[scene_id] = []
            scenes[scene_id].append(obj)
        assert idxs.size(0) == pred.shape[0]
        count += idxs.size(0)
        print('%d / %d objects processed' % (count, len(test_loader.dataset)))
    print(scenes)
    return scenes

def get_inference_options():
    from argparse import Namespace
    opt = Namespace(
        run_dir='/home/michelle/outputs/enjoy_large_box',
        dataset='clevr_dart',
        load_checkpoint_path='/home/michelle/outputs/placing_v4_direct/xyz_up/checkpoint_best.pt',
        gpu_ids=[0],
        clevr_mini_img_dir='/home/michelle/datasets/enjoy_large_box/images',
        clevr_mini_ann_path='/home/michelle/datasets/enjoy_large_box/objects.json',
        concat_img=1,
        with_depth=0,
        split_id=0,
        batch_size=20,
        num_workers=4,
        learning_rate=0.002,
        shuffle_data=False,
        pred_attr=False,
        pred_position=True,
        pred_z_dir=True,
        pred_z_size=True
    )
    return opt


def eval_with_loader(images, masks, load_checkpoint_path, pred_format, pred_pos_format, pred_rot_format):
    infer_objects = []
    dataset = ClevrDartObjectDataset(images, masks)
    loader = DataLoader(dataset=dataset, batch_size=20, num_workers=1, shuffle=False)
    for data in loader:
        infer_objects += pred_pose(data, load_checkpoint_path, pred_format, pred_pos_format, pred_rot_format)
    return infer_objects
    

def eval(rgb, masks, load_checkpoint_path, pred_format, pred_pos_format, pred_rot_format):
    data_tensors = [get_input_data(rgb, mask) for mask in masks]
    data = torch.cat((data_tensors[0].unsqueeze(0), data_tensors[1].unsqueeze(0)), 0)
    infer_objects = pred_pose(data, load_checkpoint_path, pred_format, pred_pos_format, pred_rot_format)
    return infer_objects

def get_input_data(img, mask):
    global out_i

    transform = transforms.Compose([transforms.ToTensor()])
    seg_transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
    ]
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((320, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
    ]

    img = transform(img)

    # Mask -> xywh
    mask = np.asfortranarray(mask)
    rle = mask_util.encode(mask)
    rle['counts'] = rle['counts'].decode('ASCII')
    bbox = mask_util.toBbox(rle)    # xywh
    x, y, w, h = [int(bbox[i]) for i in (0, 1, 2, 3)]

    if mask_util.area(rle) > 0:
        seg = img[:, y:y+h, x:x+w].clone()
    else:
        # Seg transform doesn't work when bbox all zeros
        # By default the seg channels are all zeros.
        print(f'Invalid bbox: {(bbox)}')

    # Final data tensor, initialized with zeros.
    data = img.clone().resize_(6, 480, 480).fill_(0)

    # Store the segmentation.
    data[0:3, :, :] = transforms.Compose(seg_transform_list)(seg)

    # Crop the object out.
    img[:, y:y + h, x:x + w] = 0.0
    data[3:6, 80:400, :] = transforms.Compose(transform_list)(img)

    return data

def pred_pose(data, load_checkpoint_path, pred_format, pred_pos_format, pred_rot_format):
    model = get_model(
        concat_img=1,
        with_depth=0,
        load_checkpoint_path=load_checkpoint_path,
        dataset='clevr_dart',
        with_rot=0,
        learning_rate=0.002,
        gpu_ids=[0]
    )
    for out_i in range(len(data)):
        os.makedirs('/home/michelle/outputs/tmp', exist_ok=True)
        imageio.imwrite(f'/home/michelle/outputs/tmp/{out_i}.png', (data[out_i, 3:6, :, :].permute(1, 2, 0).numpy()*0.225+0.5) * 255)
        imageio.imwrite(f'/home/michelle/outputs/tmp/{out_i}_seg.png', (data[out_i, 0:3, :, :].permute(1, 2, 0).numpy()*0.225+0.5) * 255)
    model.set_input(data)
    model.forward()
    pred = model.get_pred()
    infer_objects = []
    assert len(pred) == 2
    for i in range(len(pred)):
        feat_vec = pred[i]
        print(feat_vec)
        pred_position = feat_vec[0:3].tolist()
        up_vector = feat_vec[3:6].tolist()

        # print(feat_vec)

        # # Verify that the number of output features is as expected.
        # format2feats = {
        #     'attr_xy': 12,
        #     'xyz_up': 7,
        #     'attr_xyz_up': 16,
        # }
        # assert len(feat_vec) == format2feats[pred_format]

        # # Construct the model's pose predictions.    
        # pred_position = [0.0, 0.0, 0.0]
        # pred_rotation = np.eye(3)
        # if pred_format == 'attr_xy':
        #     pred_position[:2] = feat_vec[9:11].tolist()
        # elif pred_format == 'xyz_up':
        #     pred_position = feat_vec[:3].tolist()
        #     up_vector = feat_vec[3:6].tolist()
        #     pred_rotation[:, -1] = up_vector
        # elif pred_format == 'attr_xyz_up':
        #     raise NotImplementedError
        # else:
        #     raise ValueError(f'Unsupported prediction format: {self.pred_format}.')
        
        # # Select various elements of predictions to use as the pose input.
        # assert pred_pos_format in ['xy', 'xyz']
        # assert pred_rot_format in ['identity', 'up']
        # position = [0.0, 0.0, 0.0]
        # rotation = np.eye(3)
        # if pred_pos_format == 'xy':
        #     position[:2] = pred_position[:2]
        # elif pred_pos_format == 'xyz':
        #     position = pred_position
        
        # if pred_rot_format == 'identity':
        #     pass
        # elif pred_rot_format == 'up':
        #     rotation = pred_rotation
        
        # # Store the final pose.
        # rotation = list(rotation.reshape((9,)))
        # pose = (position, rotation)
        infer_objects.append({
            'position': pred_position,
            'up_vector': up_vector,
            'feat_vec': feat_vec
        })

    return infer_objects