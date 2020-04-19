STATES_DIR=~/mguo/data/states
PARTIAL_STATES_DIR=$STATES_DIR/partial
FULL_STATES_DIR=$STATES_DIR/full


time python ns_vqa_dart/bullet/states/sample_states.py \
    --seed 1 \
    --src_dir $PARTIAL_STATES_DIR/stacking_v003_2K \
    --dst_dir $PARTIAL_STATES_DIR/stacking_v003_2K_20K \
    --sample_size 20000

time python ns_vqa_dart/bullet/states/complete_states.py \
    --src_dir $PARTIAL_STATES_DIR/stacking_v003_2K_20K \
    --dst_dir $FULL_STATES_DIR/stacking_v003_2K_20K

time python ns_vqa_dart/bullet/states/add_surrounding_states.py \
    --src_dir $FULL_STATES_DIR/stacking_v003_2K_20K \
    --dst_dir $FULL_STATES_DIR/stacking_v003_2K_20K

cd $FULL_STATES_DIR
time zip -r stacking_v003_2K_20K.zip stacking_v003_2K_20K
