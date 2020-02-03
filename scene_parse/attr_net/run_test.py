import os
import json
import numpy as np

from options import get_options
from datasets import get_dataloader
from model import get_model

# COMP_CAT_DICT_PATH = 'tools/clevr_comp_cat_dict.json'

# def get_feat_vec_clevr_dart(obj):
#     attr_to_idx = {
#         'box': 0,
#         'cylinder': 1,
#         'sphere': 2,
#         'small': 3,
#         'large': 4,
#         'red': 5,
#         'orange': 6,
#         'green': 7,
#         'blue': 8
#     }
#     feat_vec = np.zeros(9+3+9)
#     for attr in ['color', 'shape', 'size']:
#         feat_vec[attr_to_idx[obj[attr]]] = 1
#     feat_vec[9:12] = obj['position']
#
#     feat_vec[12:15] = obj['rotation_x'] - feat_vec[9:12]
#     feat_vec[15:18] = obj['rotation_y'] - feat_vec[9:12]
#     feat_vec[18:21] = obj['rotation_z'] - feat_vec[9:12]
#
#     return list(feat_vec)

def get_attrs_clevr_dart(opt, feat_vec):
    attr2choices = {
        'shape': ['box', 'cylinder', 'sphere'],
        'size': ['small', 'large'],
        'color': ['red', 'yellow', 'green', 'blue']
    }
    
    start_idx = 0
    obj = {}
    if opt.pred_attr:
        for attr in ['color', 'shape', 'size']:
            end_idx = start_idx + len(attr2choices[attr])
            pred_idx = np.argmax(feat_vec[start_idx:end_idx])
            obj[attr] = attr2choices[attr][pred_idx]
            start_idx = end_idx
    if opt.pred_position:
        end_idx = start_idx + 3
        obj['position'] = feat_vec[start_idx:end_idx].tolist()
        start_idx = end_idx
    if opt.pred_z_dir:
        end_idx = start_idx + 3
        obj['z_dir'] = feat_vec[start_idx:end_idx].tolist()
        start_idx = end_idx
    if opt.pred_z_size:
        obj['z_size'] = float(feat_vec[start_idx])
    
    return obj


opt = get_options('test')
test_loader = get_dataloader(opt, 'test')
model = get_model(opt)

# if opt.use_cat_label:
#     with open(COMP_CAT_DICT_PATH) as f:
#         cat_dict = utils.invert_dict(json.load(f))

scenes = [{
    'image_index': i,
    'image_filename': 's%05d.png' % (i+opt.split_id),
    'objects': []
} for i in range(2000)]     # TODO

count = 0
for data, _, idxs, cat_idxs in test_loader:
    # each data should be one obj and its scene
    model.set_input(data)
    model.forward()
    pred = model.get_pred()
    for i in range(pred.shape[0]):  # batchsize?
        img_id = idxs[i] - opt.split_id
        obj = get_attrs_clevr_dart(opt, pred[i])
        scenes[img_id]['objects'].append(obj)
    assert idxs.size(0) == pred.shape[0]
    count += idxs.size(0)
    print('%d / %d objects processed' % (count, len(test_loader.dataset)))

output = {
    'info': '%s derendered scene' % opt.dataset,
    'scenes': scenes,
}
print('| saving annotation file to %s' % opt.output_path)
# utils.mkdirs(os.path.dirname(opt.output_path))
with open(opt.output_path, 'w') as fout:
    json.dump(output, fout, sort_keys=True, indent=2, separators=(',', ': '))