ROOT_DIR=~/mguo
STATES_SET=dash_v004_20K
IMG_SET=dash_v004_20K
CAM_SET=dash_v004_20K
DATA_SET=dash_v004_20K
OUTPUT_SET=dash_v004_20K

DATA_DIR=$ROOT_DIR/data/datasets/$DATA_SET/data
STATES_DIR=$ROOT_DIR/data/states/full/$STATES_SET
IMG_DIR=$ROOT_DIR/data/datasets/$IMG_SET/unity_output/images
CAM_DIR=$ROOT_DIR/data/datasets/$CAM_SET/unity_output/json

RUN_DIR=$ROOT_DIR/outputs/$OUTPUT_SET
PRED_PATH=$RUN_DIR/pred_train.json
HTML_DIR=$ROOT_DIR/html/$OUTPUT_SET

EVAL_START=0
EVAL_END=16000
COORD_FRAME=unity_camera
HTML_N_SCENES=30

time python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --eval_start_id $EVAL_START \
    --eval_end_id $EVAL_END \
    --output_path $PRED_PATH \
    --load_checkpoint_path $RUN_DIR/checkpoint_best.pt \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --num_workers 8
