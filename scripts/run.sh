cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v009 \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/ego_v009 \
    --start_id 0 \
    --end_id 19999 \
    --checkpoint_every 2000 \
    --num_iters 20 \
    --height 480 \
    --width 480 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v009 \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/ego_v009 \
    --output_path ~/outputs/ego_v009/test.json \
    --load_checkpoint_path ~/outputs/ego_v009/checkpoint_best.pt \
    --start_id 20000 \
    --end_id 21999 \
    --height 480 \
    --width 480 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8
