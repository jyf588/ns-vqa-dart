cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v002 \
    --dataset_dir ~/datasets/ego_v002 \
    --output_path ~/outputs/ego_v002/test.json \
    --load_checkpoint_path ~/outputs/ego_v002/checkpoint_best.pt \
    --split_id 20000 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height
