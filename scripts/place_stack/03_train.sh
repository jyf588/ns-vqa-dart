MODELS_GROUP=place_stack
DATA_DIRS_JSON=ns_vqa_dart/scripts/$MODELS_GROUP/data_dirs.json
RUN_DIR=/home/mguo/outputs/$MODELS_GROUP
CHECKPOINT_EVERY=7500
NUM_ITERS=600000


python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --data_dirs_json $DATA_DIRS_JSON \
    --run_dir $RUN_DIR \
    --checkpoint_every $CHECKPOINT_EVERY \
    --num_iters $NUM_ITERS \
    --num_workers 8
