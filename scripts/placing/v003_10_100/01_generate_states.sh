ROOT_DIR=/media/sdc3/mguo
STATES_DIR=$ROOT_DIR/data/states
PARTIAL_STATES_DIR=$STATES_DIR/partial
FULL_STATES_DIR=$STATES_DIR/full

time python enjoy.py \
    --env-name InmoovHandPlaceBulletEnv-v9 \
    --load-dir trained_models_0404_0_n_place_0404_0/ppo \
    --non-det 0 \
    --seed=18980 \
    --random_top_shape 1 \
    --renders 0 \
    --exclude_hard 0 \
    --obs_noise 1 \
    --n_best_cand 2 \
    --cotrain_stack_place 0 \
    --place_floor 1 \
    --grasp_pi_name "0404_0_n_20_40" \
    --use_obj_heights 1 \
    --save_states 1 \
    --states_dir $PARTIAL_STATES_DIR/placing_v003_10 \
    --n_trials 10

time python ns_vqa_dart/bullet/states/sample_states.py \
    --seed 1 \
    --src_dir $PARTIAL_STATES_DIR/placing_v003_10 \
    --dst_dir $PARTIAL_STATES_DIR/placing_v003_10_100 \
    --sample_size 100

time python ns_vqa_dart/bullet/states/complete_states.py \
    --src_dir $PARTIAL_STATES_DIR/placing_v003_10_100 \
    --dst_dir $FULL_STATES_DIR/placing_v003_10_100_nosur

time python ns_vqa_dart/bullet/states/add_surrounding_states.py \
    --src_dir $FULL_STATES_DIR/placing_v003_10_100_nosur \
    --dst_dir $FULL_STATES_DIR/placing_v003_10_100

cd $FULL_STATES_DIR
rm placing_v003_10_100.zip
time zip -r placing_v003_10_100.zip placing_v003_10_100
