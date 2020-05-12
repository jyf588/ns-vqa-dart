TRAIN_SET=seg_tiny
VAL_SET=seg_tiny

ROOT_DIR=/home/mguo
RUN_DIR=$ROOT_DIR/outputs/attr_net/$TRAIN_SET

COORD_FRAME=unity_camera

CHECKPOINT_EVERY=1000
NUM_ITERS=600000


time python ns_vqa_dart/scene_parse/attr_net/run_train.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --train_set $TRAIN_SET \
    --eval_set $VAL_SET \
    --checkpoint_every $CHECKPOINT_EVERY \
    --num_iters $NUM_ITERS \
    --num_workers 8 \
    --coordinate_frame $COORD_FRAME
