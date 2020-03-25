ROOT_DIR=/media/michelle/68B62784B62751BC
SET=dash_v001_100
DATA_DIR=$ROOT_DIR/data/datasets/$SET/data
RUN_DIR=$ROOT_DIR/outputs/$SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=$ROOT_DIR/html/$SET
IMG_DIR=$ROOT_DIR/data/datasets/$SET/images

CHECKPOINT_EVERY=2000
NUM_ITERS=60000
COORD_FRAME=world
HTML_N_OBJECTS=2

time python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --dataset_dir $DATA_DIR \
    --output_path $PRED_PATH \
    --load_checkpoint_path $RUN_DIR/checkpoint_best.pt \
    --coordinate_frame $COORD_FRAME \
    --num_workers 8

python ns_vqa_dart/bullet/html_images.py \
    --dataset_dir $DATA_DIR \
    --img_dir $IMG_DIR \
	--pred_path $PRED_PATH \
    --html_dir $HTML_DIR \
    --coordinate_frame $COORD_FRAME \
    --n_objects $HTML_N_OBJECTS

python ns_vqa_dart/bullet/html.py \
    --html_dir $HTML_DIR \
    --show_objects
