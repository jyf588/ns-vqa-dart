cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v003 \
    --dataset_dir ~/datasets/ego_v003 \
    --split_id 20000 \
    --checkpoint_every 200 \
    --num_iters 60000 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height
