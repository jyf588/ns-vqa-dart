STATES_DIR=/home/mguo/states
PARTIAL_STATES_DIR=$STATES_DIR/partial
FULL_STATES_DIR=$STATES_DIR/full



time python ns_vqa_dart/bullet/states/sample_states.py \
    --seed 1 \
    --src_dir $PARTIAL_STATES_DIR/place_100K_0517 \
    --dst_dir $PARTIAL_STATES_DIR/place_2K_20K_0517 \
    --start_trial 1 \
    --end_trial 2001 \
    --sample_size 20000

