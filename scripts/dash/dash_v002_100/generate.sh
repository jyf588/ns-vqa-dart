ROOT_DIR=/media/michelle/68B62784B62751BC
STATES_SET=dash_v001
IMG_SET=dash_v002
DST_SET=dash_v002_100

python ns_vqa_dart/bullet/gen_dataset.py \
    --states_dir $ROOT_DIR/data/states/full/$STATES_SET \
    --img_dir $ROOT_DIR/data/datasets/$IMG_SET/images \
    --dst_dir $ROOT_DIR/data/datasets/$DST_SET/data \
    --start_sid 21900 \
    --end_sid 22000
