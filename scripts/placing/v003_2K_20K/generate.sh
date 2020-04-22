ROOT_DIR=/media/sdc3/mguo
SRC_SET=placing_v003_2K_20K
DST_SET=placing_v003_2K_20K

time python ns_vqa_dart/bullet/gen_dataset.py \
    --states_dir $ROOT_DIR/data/states/full/$SRC_SET \
    --img_dir $ROOT_DIR/data/datasets/$SRC_SET/unity_output/images \
    --cam_dir $ROOT_DIR/data/datasets/$SRC_SET/unity_output/json \
    --dst_dir $ROOT_DIR/data/datasets/$DST_SET/data \
    --camera_control stack \
    --coordinate_frame unity_camera \
    --start_sid 0 \
    --end_sid 20000
