cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v004 \
    --dataset_dir ~/datasets/ego_v004 \
    --output_path ~/outputs/ego_v004/test.json \
    --load_checkpoint_path ~/outputs/ego_v004/checkpoint_best.pt \
    --split_id 20000 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height

cd ../..

time python bullet/metrics.py \
    --dataset_dir ~/datasets/ego_v004 \
	--pred_path ~/outputs/ego_v004/test.json

time python bullet/visualize.py \
    --dataset_dir ~/datasets/ego_v004 \
    --pred_path ~/outputs/ego_v004/test.json \
    --output_dir ~/analysis/ego_v004