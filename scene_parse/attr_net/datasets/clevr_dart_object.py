import os
import json

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pycocotools.mask as mask_util


class ClevrDartObjectDataset(Dataset):

    def __init__(self, obj_ann_path, img_dir, split,
                 min_img_id=None, max_img_id=None, concat_img=True, with_depth=False, with_rot=True):
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
        self.cat_ids = anns['category_idxs'][min_id: max_id]
        if anns['feature_vectors'] != []:
            self.feat_vecs = np.array(anns['feature_vectors'][min_id: max_id]).astype(float)
        else:
            self.feat_vecs = None

        self.img_dir = img_dir
        self.split = split
        self.concat_img = concat_img
        self.with_depth = with_depth
        self.with_rot = with_rot

        transform_list = [transforms.ToTensor()]    # toTensor add dimension to gray image
        self._transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = 's%04d.png' % (self.img_ids[idx])
        # print(os.path.join(self.img_dir, img_name))
        # input("press enter")
        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_COLOR)
        img = self._transform(img)

        if self.with_depth:
            dimg_name = 's%04d_depth.png' % (self.img_ids[idx])
            dimg = cv2.imread(os.path.join(self.img_dir, dimg_name), cv2.IMREAD_GRAYSCALE)
            # print(dimg.shape)
            dimg = self._transform(dimg)
            # print(dimg.size())

        label = -1
        if self.feat_vecs is not None:
            label = torch.Tensor(self.feat_vecs[idx])
            if not self.with_rot:
                label = label[:12]  # TODO
        img_id = self.img_ids[idx]
        cat_id = self.cat_ids[idx]

        mask = torch.Tensor(mask_util.decode(self.obj_masks[idx]))
        seg = img.clone()
        for i in range(3):
            seg[i, :, :] = img[i, :, :] * mask.float()

        if self.with_depth:
            dseg = dimg.clone()
            dseg[0, :, :] = dimg[0, :, :] * mask.float()
            dtransform_list = [transforms.ToPILImage(),
                              transforms.Resize((149, 224)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5], std=[0.3])]      # TODO: maybe changed to 0.6

        transform_list = [transforms.ToPILImage(),
                          transforms.Resize((149, 224)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if self.concat_img:
            if self.with_depth:
                data = img.clone().resize_(8, 224, 224).fill_(0)
                data[0:3, 38:187, :] = transforms.Compose(transform_list)(seg)
                data[3:4, 38:187, :] = transforms.Compose(dtransform_list)(dseg)
                data[4:7, 38:187, :] = transforms.Compose(transform_list)(img)
                data[7:8, 38:187, :] = transforms.Compose(dtransform_list)(dimg)
                # print(dimg.mean())
                # print("bb", dimg.std())
            else:
                data = img.clone().resize_(6, 224, 224).fill_(0)        # TODO: what is this?
                data[0:3, 38:187, :] = transforms.Compose(transform_list)(seg)
                data[3:6, 38:187, :] = transforms.Compose(transform_list)(img)
        else:
            data = img.clone().resize_(3, 224, 224).fill_(0)
            data[:, 38:187, :] = transforms.Compose(transform_list)(seg)

        return data, label, img_id, cat_id