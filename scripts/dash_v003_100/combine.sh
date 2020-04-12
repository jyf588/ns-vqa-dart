ROOT_DIR=~/mguo/data

time python ns_vqa_dart/bullet/combine_states.py \
    --src_dirs \
        $ROOT_DIR/states/full/planning_v003 \
        $ROOT_DIR/states/full/placing_surround \
    --dst_dir $ROOT_DIR/states/full/dash_v003_100 \
    --n_states 100

cd ~/mguo/data/states/full
time zip -r dash_v003_100.zip dash_v003_100
