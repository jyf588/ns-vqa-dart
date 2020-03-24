SET=dash_v001
DATA_DIR=~/datasets/$SET
RUN_DIR=~/outputs/$SET
PRED_PATH=$RUN_DIR/pred.json
HTML_DIR=~/html/$SET

CHECKPOINT_EVERY=2000
NUM_ITERS=100
COORD_FRAME=world
HTML_N_OBJECTS=3

python ns_vqa_dart/bullet/html_images.py \
    --dataset_dir $DATA_DIR \
	--pred_path $PRED_PATH \
    --html_dir $HTML_DIR \
    --coordinate_frame $COORD_FRAME \
    --n_objects $HTML_N_OBJECTS

python ns_vqa_dart/bullet/html.py \
    --html_dir $HTML_DIR \
    --show_objects
    