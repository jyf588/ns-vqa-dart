TRAIN_SET=seg_tiny
VAL_SET=seg_tiny

ROOT_DIR=/home/mguo
RUN_DIR=$ROOT_DIR/outputs/attr_net/$TRAIN_SET
PRED_PATH=$RUN_DIR/pred.json

COORD_FRAME=unity_camera

time python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --dataset dash \
    --run_dir $RUN_DIR \
    --eval_set $VAL_SET \
    --output_path $PRED_PATH \
    --load_checkpoint_path $RUN_DIR/checkpoint_best.pt \
    --coordinate_frame $COORD_FRAME \
    --num_workers 8
