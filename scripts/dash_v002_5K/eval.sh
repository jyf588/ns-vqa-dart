ROOT_DIR=~/mguo
SET=dash_v002_5K
DATA_DIR=$ROOT_DIR/data/datasets/$SET/data
RUN_DIR=$ROOT_DIR/outputs/$SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=$ROOT_DIR/html/$SET
IMG_DIR=$ROOT_DIR/data/datasets/$SET/images
CAM_DIR=$ROOT_DIR/data/datasets/$SET/camera

CHECKPOINT_EVERY=2000
NUM_ITERS=60000
COORD_FRAME=camera
HTML_N_SCENES=30


time python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
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
