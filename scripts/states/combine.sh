ROOT_DIR=/media/michelle/68B62784B62751BC/mguo/data

time python ns_vqa_dart/bullet/combine_states.py \
    --src_dirs \
        $ROOT_DIR/states/full/planning_v003 \
        $ROOT_DIR/states/full/stacking_box \
        $ROOT_DIR/states/full/stacking_cyl \
    --dst_dir $ROOT_DIR/states/full/dash_v002 \
    --n_states 100
