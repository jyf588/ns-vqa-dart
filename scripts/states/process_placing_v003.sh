ROOT_DIR=~/mguo/data

time python ns_vqa_dart/bullet/complete_states.py \
    --src_dir $ROOT_DIR/states/partial/placing_v003 \
    --dst_dir $ROOT_DIR/states/full/placing_v003


time python ns_vqa_dart/bullet/add_surrounding_states.py \
    --src_dir $ROOT_DIR/states/full/placing_v003 \
    --dst_dir $ROOT_DIR/states/full/placing_v003_surround
