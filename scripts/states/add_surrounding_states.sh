ROOT_DIR=/media/michelle/68B62784B62751BC/data


time python ns_vqa_dart/bullet/add_surrounding_states.py \
    --src_dir $ROOT_DIR/states/full/delay_box_states \
    --dst_dir $ROOT_DIR/states/full/stacking_box

time python ns_vqa_dart/bullet/add_surrounding_states.py \
    --src_dir $ROOT_DIR/states/full/delay_cyl_states \
    --dst_dir $ROOT_DIR/states/full/stacking_cyl
