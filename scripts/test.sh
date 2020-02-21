cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v006 \
    --dataset_dir ~/datasets/ego_v006 \
    --output_path ~/outputs/ego_v006/test.json \
    --load_checkpoint_path ~/outputs/ego_v006/checkpoint_best.pt \
    --split_id 20000 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --batch_size 20 \
    --num_workers 8
