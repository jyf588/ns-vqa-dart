ROOT_DIR=~/mguo
SET=dash_v002
DATA_DIR=$ROOT_DIR/data/datasets/$SET/data
RUN_DIR=$ROOT_DIR/outputs/$SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=$ROOT_DIR/html/$SET
IMG_DIR=$ROOT_DIR/data/datasets/$SET/images

CHECKPOINT_EVERY=2000
NUM_ITERS=100
COORD_FRAME=world
HTML_N_OBJECTS=100



python ns_vqa_dart/bullet/html.py \
    --html_dir $HTML_DIR \
    --show_objects
