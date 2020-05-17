time python ns_vqa_dart/bullet/states/sample_states.py \
    --seed 1 \
    --src_dir $PARTIAL_STATES_DIR/placing_v003_2K \
    --dst_dir $PARTIAL_STATES_DIR/placing_v003_2K_20K \
    --sample_size 20000

time python ns_vqa_dart/bullet/states/complete_states.py \
    --src_dir $PARTIAL_STATES_DIR/placing_v003_2K_20K \
    --dst_dir $FULL_STATES_DIR/placing_v003_2K_20K_nosur

time python ns_vqa_dart/bullet/states/add_surrounding_states.py \
    --src_dir $FULL_STATES_DIR/placing_v003_2K_20K_nosur \
    --dst_dir $FULL_STATES_DIR/placing_v003_2K_20K

cd $FULL_STATES_DIR
rm placing_v003_2K_20K.zip
time zip -r placing_v003_2K_20K.zip placing_v003_2K_20K
