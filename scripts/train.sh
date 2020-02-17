cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v005 \
    --dataset_dir ~/datasets/ego_v005 \
    --split_id 20000 \
    --checkpoint_every 2000 \
    --num_iters 60000 \
    --batch_size 20 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height \
    --coordinate_frame camera
