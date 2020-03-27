ROOT_DIR=/media/michelle/68B62784B62751BC
SRC_SET=dash_v001
DST_SET=dash_v002

python ns_vqa_dart/bullet/gen_dataset.py \
    --states_dir $ROOT_DIR/data/states/full/$SRC_SET \
    --img_dir $ROOT_DIR/data/datasets/$DST_SET/images \
    --dst_dir $ROOT_DIR/data/datasets/$DST_SET/data \
    --start_sid 0 \
    --end_sid 22000
