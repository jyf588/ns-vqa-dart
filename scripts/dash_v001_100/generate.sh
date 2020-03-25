ROOT_DIR=/media/michelle/68B62784B62751BC

python ns_vqa_dart/bullet/gen_dataset.py \
    --states_dir $ROOT_DIR/data/states/full/dash_v001 \
    --img_dir $ROOT_DIR/data/datasets/dash_v001_100/images \
    --dst_dir $ROOT_DIR/data/datasets/dash_v001_100/data \
    --start_sid 0 \
    --end_sid 100
