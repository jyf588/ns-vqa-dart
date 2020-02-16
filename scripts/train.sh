cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_001 \
    --dataset_dir ~/datasets/ego_001 \
    --split_id 0 \
    --checkpoint_every 20 \
    --num_iters 60000 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height
