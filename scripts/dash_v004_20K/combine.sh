ROOT_DIR=~/mguo/data

time python ns_vqa_dart/bullet/combine_states.py \
    --src_dirs \
        $ROOT_DIR/states/full/planning_v003 \
        $ROOT_DIR/states/full/placing_v002_surround \
    --dst_dir $ROOT_DIR/states/full/dash_v004_20K \
    --n_states 20000
