cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v003 \
    --dataset_dir ~/datasets/ego_v003 \
    --split_id 0 \
    --checkpoint_every 200 \
    --num_iters 20 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v003 \
    --dataset_dir ~/datasets/ego_v003 \
    --output_path ~/outputs/ego_v003/test.json \
    --load_checkpoint_path ~/outputs/ego_v003/checkpoint.pt \
    --split_id 0 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height

cd ../..

python bullet/metrics.py \
    --dataset_dir ~/datasets/ego_v003 \
	--pred_path ~/outputs/ego_v003/test.json

python bullet/visualize.py \
    --dataset_dir ~/datasets/ego_v003 \
    --pred_path ~/outputs/ego_v003/test.json \
    --output_dir ~/analysis/ego_v003