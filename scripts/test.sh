cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_001 \
    --dataset_dir ~/datasets/ego_001 \
    --output_path ~/outputs/ego_001/test.json \
    --load_checkpoint_path ~/outputs/ego_001/checkpoint_best.pt \
    --split_id 0 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height
