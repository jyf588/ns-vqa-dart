ROOT_DIR=~/mguo/data

time python ns_vqa_dart/bullet/combine_states.py \
    --partition_path ns_vqa_dart/scripts/dash_v005_20K/partition.json \
    --dst_dir $ROOT_DIR/states/full/dash_v005_20K \
    --n_states 20000
