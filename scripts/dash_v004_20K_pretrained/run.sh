ROOT_DIR=~/mguo
STATES_SET=dash_v004_20K
IMG_SET=dash_v004_20K
CAM_SET=dash_v004_20K
DATA_SET=dash_v004_20K
OUTPUT_SET=dash_v004_20K_pretrained

DATA_DIR=$ROOT_DIR/data/datasets/$DATA_SET/data
STATES_DIR=$ROOT_DIR/data/states/full/$STATES_SET
IMG_DIR=$ROOT_DIR/data/datasets/$IMG_SET/unity_output/images
CAM_DIR=$ROOT_DIR/data/datasets/$CAM_SET/unity_output/json

RUN_DIR=$ROOT_DIR/outputs/$OUTPUT_SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=$ROOT_DIR/html/$OUTPUT_SET

CHECKPOINT_EVERY=2000
NUM_ITERS=60000
TRAIN_START=0
TRAIN_END=16000
EVAL_START=0
EVAL_END=4000
COORD_FRAME=unity_camera
HTML_N_SCENES=30

time python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --train_start_id $TRAIN_START \
    --train_end_id $TRAIN_END \
    --eval_start_id $EVAL_START \
    --eval_end_id $EVAL_END \
    --checkpoint_every $CHECKPOINT_EVERY \
    --num_iters $NUM_ITERS \
    --num_workers 8

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

python ns_vqa_dart/bullet/html_images.py \
    --dataset_dir $DATA_DIR \
    --states_dir $STATES_DIR \
    --img_dir $IMG_DIR \
	--pred_path $PRED_PATH \
    --html_dir $HTML_DIR \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --n_scenes $HTML_N_SCENES


python ns_vqa_dart/bullet/html.py \
    --html_dir $HTML_DIR \
    --show_objects
