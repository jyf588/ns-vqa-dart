ROOT_DIR=~/mguo
SRC_SET=dash_v002_5K
DST_SET=dash_v002_100

time python ns_vqa_dart/bullet/gen_dataset.py \
    --states_dir $ROOT_DIR/data/states/full/$SRC_SET \
    --img_dir $ROOT_DIR/data/datasets/$DST_SET/images \
    --cam_dir $ROOT_DIR/data/datasets/$DST_SET/camera \
    --dst_dir $ROOT_DIR/data/datasets/$DST_SET/data \
    --coordinate_frame camera \
    --start_sid 0 \
    --end_sid 100
