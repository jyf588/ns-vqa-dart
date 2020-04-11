ROOT_DIR=~/mguo
IMG_SET=dash_v002_5K
CAM_SET=dash_v002_5K
DATA_SET=dash_v003_5K
OUTPUT_SET=dash_v003_5K

DATA_DIR=$ROOT_DIR/data/datasets/$DATA_SET/data
IMG_DIR=$ROOT_DIR/data/datasets/$IMG_SET/images
CAM_DIR=$ROOT_DIR/data/datasets/$CAM_SET/camera

RUN_DIR=$ROOT_DIR/outputs/$OUTPUT_SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=$ROOT_DIR/html/$OUTPUT_SET

CHECKPOINT_EVERY=2000
NUM_ITERS=100
START_ID=0
SPLIT_ID=4000
END_ID=5000
COORD_FRAME=unity_camera
HTML_N_SCENES=10

time python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --train_start_id $START_ID \
    --train_end_id $SPLIT_ID \
    --eval_start_id $SPLIT_ID \
    --eval_end_id $END_ID \
    --checkpoint_every $CHECKPOINT_EVERY \
    --num_iters $NUM_ITERS \
    --num_workers 8

time python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --eval_start_id $SPLIT_ID \
    --eval_end_id $END_ID \
    --output_path $PRED_PATH \
    --load_checkpoint_path $RUN_DIR/checkpoint_best.pt \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --num_workers 8

python ns_vqa_dart/bullet/html_images.py \
    --dataset_dir $DATA_DIR \
    --img_dir $IMG_DIR \
	--pred_path $PRED_PATH \
    --html_dir $HTML_DIR \
    --coordinate_frame $COORD_FRAME \
    --cam_dir $CAM_DIR \
    --n_scenes $HTML_N_SCENES


python ns_vqa_dart/bullet/html.py \
    --html_dir $HTML_DIR \
    --show_objects
