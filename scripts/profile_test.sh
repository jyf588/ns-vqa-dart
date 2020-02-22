cd scene_parse/attr_net

time python run_test_profiler.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v008 \
    --dataset_dir ~/datasets/ego_v008 \
    --output_path ~/outputs/ego_v008/test.json \
    --load_checkpoint_path ~/outputs/ego_v008/checkpoint_best.pt \
    --split_id 20000 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --batch_size 7 \
    --num_workers 8
