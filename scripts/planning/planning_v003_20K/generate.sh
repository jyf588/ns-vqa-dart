ROOT_DIR=~/mguo
SRC_SET=planning_v003_20K
DST_SET=planning_v003_20K

time python ns_vqa_dart/bullet/gen_dataset.py \
    --states_dir $ROOT_DIR/data/states/full/$SRC_SET \
    --img_dir $ROOT_DIR/data/datasets/$SRC_SET/unity_output/images \
    --cam_dir $ROOT_DIR/data/datasets/$SRC_SET/unity_output/json \
    --dst_dir $ROOT_DIR/data/datasets/$DST_SET/data \
    --camera_control center \
    --coordinate_frame unity_camera \
    --start_sid 16000 \
    --end_sid 16100
