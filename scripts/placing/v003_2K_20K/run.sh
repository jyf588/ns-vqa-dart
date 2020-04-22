ROOT_DIR=/media/sdc3/mguo
STATES_SET=placing_v003_2K_20K
IMG_SET=placing_v003_2K_20K
CAM_SET=placing_v003_2K_20K
DATA_SET=placing_v003_2K_20K
OUTPUT_SET=placing_v003_2K_20K

DATA_DIR=$ROOT_DIR/data/datasets/$DATA_SET/data
STATES_DIR=$ROOT_DIR/data/states/full/$STATES_SET
IMG_DIR=$ROOT_DIR/data/datasets/$IMG_SET/unity_output/images
CAM_DIR=$ROOT_DIR/data/datasets/$CAM_SET/unity_output/json

RUN_DIR=$ROOT_DIR/outputs/$OUTPUT_SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=$ROOT_DIR/html/$OUTPUT_SET
TRAIN_PLOT_PATH=$RUN_DIR/plots/train.png
VAL_PLOT_PATH=$RUN_DIR/plots/val.png

CHECKPOINT_EVERY=10000
NUM_ITERS=600000
TRAIN_START=0
TRAIN_END=16000
VAL_START=16000
VAL_END=20000
CAMERA_CONTROL=stack
COORD_FRAME=unity_camera
HTML_N_SCENES=30

time python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --train_start_id $TRAIN_START \
    --train_end_id $TRAIN_END \
    --eval_start_id $VAL_START \
    --eval_end_id $VAL_END \
    --checkpoint_every $CHECKPOINT_EVERY \
    --num_iters $NUM_ITERS \
    --num_workers 8

time python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --eval_start_id $TRAIN_START \
    --eval_end_id $TRAIN_END \
    --output_path $PRED_PATH \
    --load_checkpoint_path $RUN_DIR/checkpoint_best.pt \
    --camera_control $CAMERA_CONTROL \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --plot_path $TRAIN_PLOT_PATH \
    --num_workers 8

time python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --eval_start_id $VAL_START \
    --eval_end_id $VAL_END \
    --output_path $PRED_PATH \
    --load_checkpoint_path $RUN_DIR/checkpoint_best.pt \
    --camera_control $CAMERA_CONTROL \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --plot_path $VAL_PLOT_PATH \
    --num_workers 8

python ns_vqa_dart/bullet/html_images.py \
    --dataset_dir $DATA_DIR \
    --states_dir $STATES_DIR \
    --img_dir $IMG_DIR \
	--pred_path $PRED_PATH \
    --html_dir $HTML_DIR \
    --camera_control $CAMERA_CONTROL \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --n_scenes $HTML_N_SCENES

python ns_vqa_dart/bullet/html.py \
    --html_dir $HTML_DIR \
    --show_objects
