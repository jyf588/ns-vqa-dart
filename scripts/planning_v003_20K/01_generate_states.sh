STATES_DIR=~/mguo/data/states/full

python ns_vqa_dart/bullet/generate_planning_states.py \
    --output_dir $STATES_DIR/planning_v003_20K \
    --n_examples 20000


cd $STATES_DIR
time zip -r planning_v003_20K.zip planning_v003_20K
