ROOT_DIR=~/mguo
SRC_SET=dash_v002
DST_SET=dash_v002

time python ns_vqa_dart/bullet/gen_dataset.py \
    --states_dir $ROOT_DIR/data/states/full/$SRC_SET \
    --img_dir $ROOT_DIR/data/datasets/$DST_SET/images \
    --dst_dir $ROOT_DIR/data/datasets/$DST_SET/data \
    --start_sid 0 \
    --end_sid 100
