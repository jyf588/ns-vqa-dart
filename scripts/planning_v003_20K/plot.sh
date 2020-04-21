ROOT_DIR=~/mguo
STATES_SET=planning_v003_20K
IMG_SET=planning_v003_20K
CAM_SET=planning_v003_20K
DATA_SET=planning_v003_20K
OUTPUT_SET=planning_v003_20K

DATA_DIR=$ROOT_DIR/data/datasets/$DATA_SET/data
STATES_DIR=$ROOT_DIR/data/states/full/$STATES_SET
IMG_DIR=$ROOT_DIR/data/datasets/$IMG_SET/unity_output/images
CAM_DIR=$ROOT_DIR/data/datasets/$CAM_SET/unity_output/json

RUN_DIR=$ROOT_DIR/outputs/$OUTPUT_SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=$ROOT_DIR/html/$OUTPUT_SET

PLOT_PATH=$RUN_DIR/plots/val.png

CHECKPOINT_EVERY=10000
NUM_ITERS=600000
TRAIN_START=0
TRAIN_END=100
EVAL_START=16000
EVAL_END=16100
CAMERA_CONTROL=center
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
    --camera_control $CAMERA_CONTROL \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --plot_path $PLOT_PATH \
    --num_workers 8
