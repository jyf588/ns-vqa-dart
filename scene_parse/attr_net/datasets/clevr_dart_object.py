import os
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util


class ClevrDartObjectDataset(Dataset):

    def __init__(self, opt, obj_ann_path, img_dir, split,
                 min_img_id=None, max_img_id=None):
        with open(obj_ann_path) as f:
            anns = json.load(f)
        
        # search for the object id range corresponding to the image split
        min_id = 0
        if min_img_id is not None:
            while anns['image_idxs'][min_id] < min_img_id:
                min_id += 1
        max_id = len(anns['image_idxs'])
        if max_img_id is not None:
            while max_id > 0 and anns['image_idxs'][max_id - 1] >= max_img_id:
                max_id -= 1
        self.obj_masks = anns['object_masks'][min_id: max_id]
        self.img_ids = anns['image_idxs'][min_id: max_id]
        self.obj_dicts = anns['obj_dicts'][min_id: max_id]
        # if anns['feature_vectors'] != []:
        #     self.feat_vecs = np.array(anns['feature_vectors'][min_id: max_id]).astype(float)
        # else:
        #     self.feat_vecs = None

        self.img_dir = img_dir
        self.split = split
        self.opt = opt

        transform_list = [transforms.ToTensor()]    # toTensor add dimension to gray image
        self._transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = 's%05d.png' % (self.img_ids[idx])
        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)    # 0~255
        img = self._transform(img)  # 0~1

        if self.opt.with_depth:
            dimg_name = 's%04d_depth.png' % (self.img_ids[idx])
            dimg = cv2.imread(os.path.join(self.img_dir, dimg_name), cv2.IMREAD_GRAYSCALE)
            dimg = self._transform(dimg)

        # print(self.img_ids[idx])

        # label = -1
        # if self.feat_vecs is not None:
        #     label = torch.Tensor(self.feat_vecs[idx])   # TODO: feat 2 3 4 are actualy not useful
        #     # if not self.with_rot:
        #     #     label = label[:-9]  # TODO: asuume final 9 are rot

        label = self.process_labels(self.obj_dicts[idx])

        img_id = self.img_ids[idx]

        rle = self.obj_masks[idx]
        bbox = mask_util.toBbox(rle)    # xywh
        x, y, w, h = [int(bbox[i]) for i in (0, 1, 2, 3)]
        # print(x)

        seg = img[:, y:y+h, x:x+w].clone()
        seg_transform_list = [transforms.ToPILImage(),
                          transforms.Resize((480, 480)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])]
        transform_list = [transforms.ToPILImage(),
                          transforms.Resize((320, 480)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])]

        if self.opt.with_depth:
            dseg = dimg[0, y:y+h, x:x+w].clone()
            dseg_transform_list = [transforms.ToPILImage(),
                               transforms.Resize((480, 480)),
                               transforms.ToTensor(),   # back to tensor
                               transforms.Normalize(mean=[0.5], std=[0.3])]
            dtransform_list = [transforms.ToPILImage(),
                               transforms.Resize((320, 480)),
                               transforms.ToTensor(),   # back to tensor
                               transforms.Normalize(mean=[0.5], std=[0.3])]

        if self.opt.concat_img:
            if self.opt.with_depth:
                data = img.clone().resize_(8, 480, 480).fill_(0)

                data[0:3, :, :] = transforms.Compose(seg_transform_list)(seg)
                data[3:4, :, :] = transforms.Compose(dseg_transform_list)(dseg)
                # TODO: corp the object out, otherwise confusing
                img[:, y:y + h, x:x + w] = 0.0
                data[4:7, 80:400, :] = transforms.Compose(transform_list)(img)
                dimg[0, y:y + h, x:x + w] = 0.0
                data[7:8, 80:400, :] = transforms.Compose(dtransform_list)(dimg)

                # data = dimg.clone().resize_(2, 360, 360).fill_(0)
                # data[0:1, :, :] = transforms.Compose(dseg_transform_list)(dseg)
                # dimg[0, y:y + h, x:x + w] = 0.0
                # data[1:2, 60:300, :] = transforms.Compose(dtransform_list)(dimg)

                # cv2.imwrite('tmp_img_seg.png', (data[0:3, :, :].permute(1, 2, 0).numpy()*0.3+0.5) * 255)
                # cv2.imwrite('tmp_img_dseg.png', (data[3:4, :, :].permute(1, 2, 0).numpy()*0.3+0.5) * 255)
                # cv2.imwrite('tmp_img.png', (data[4:7, :, :].permute(1, 2, 0).numpy()*0.225+0.5) * 255)
                # cv2.imwrite('tmp_img_dimg.png', (data[7:8, :, :].permute(1, 2, 0).numpy()*0.225+0.5) * 255)
                # input('press enter')
            else:
                data = img.clone().resize_(6, 480, 480).fill_(0)

                data[0:3, :, :] = transforms.Compose(seg_transform_list)(seg)
                # TODO: corp the object out, otherwise confusing
                img[:, y:y + h, x:x + w] = 0.0
                data[3:6, 80:400, :] = transforms.Compose(transform_list)(img)
                # os.makedirs('tmp', exist_ok=True)
                # cv2.imwrite(f'tmp/{idx}.png', (data[3:6, :, :].permute(1, 2, 0).numpy()*0.225+0.5) * 255)
                # cv2.imwrite(f'tmp/{idx}_seg.png', (data[0:3, :, :].permute(1, 2, 0).numpy()*0.225+0.5) * 255)
                # input('press enter')
        else:
            data = img.clone().resize_(3, 480, 480).fill_(0)
            # data[:, 80:400, :] = transforms.Compose(transform_list)(seg)

        return data, label, img_id, 0       # cat_id dont care

    def process_labels(self, obj):
        """
        Create the feature vector in the following order:
            attributes (9)
            world position (3)
            world z direction vector (3)
            z size (1)
        """
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

        # Build the labels.
        feat_vec = []
        if self.opt.pred_attr:
            n_attr_feats = len(attr_to_idx)
            attr_vec = [0] * n_attr_feats
            for attr in ['color', 'shape', 'size']:
                attr_vec[attr_to_idx[obj[attr]]] = 1
            feat_vec += attr_vec
        if self.opt.pred_position:
            feat_vec += obj['world_pos']
        if self.opt.pred_z_dir:
            # Compute world z direction from rotation.
            world_z_dir = rot_to_z_dir(obj['world_rot'])
            feat_vec += world_z_dir
        if self.opt.pred_z_size:
            feat_vec.append(obj['z_size'])
        return torch.Tensor(feat_vec)

def rot_to_z_dir(world_rot):
    world_rot = np.array(world_rot).reshape((3, 3))
    z_dir = world_rot[:, -1]  # Last column
    z_dir = list(z_dir)
    return z_dir
