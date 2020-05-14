SET=placing_v003_2K_20K
MODEL_NAME=2020_04_22_04_35
DATA_DIR=/home/mguo/data/$SET/data
RUN_DIR=/home/mguo/outputs/$SET/$MODEL_NAME/eval

python ns_vqa_dart/scene_parse/attr_net/run_test.py \
    --data_dir $DATA_DIR \
    --run_dir $RUN_DIR