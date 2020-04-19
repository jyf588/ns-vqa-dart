STATES_DIR=~/mguo/data/states/full

python ns_vqa_dart/bullet/states/generate_planning_states.py \
    --output_dir $STATES_DIR/planning_v003_100 \
    --n_examples 100


cd $STATES_DIR
time zip -r planning_v003_100.zip planning_v003_100
