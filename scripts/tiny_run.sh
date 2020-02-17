python bullet/generate.py \
    --dataset_dir ~/datasets/ego_v005 \
    --n_examples 2

cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v005 \
    --dataset_dir ~/datasets/ego_v005 \
    --split_id 1 \
    --checkpoint_every 2000 \
    --num_iters 20 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height \
    --coordinate_frame camera

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v005 \
    --dataset_dir ~/datasets/ego_v005 \
    --output_path ~/outputs/ego_v005/test.json \
    --load_checkpoint_path ~/outputs/ego_v005/checkpoint.pt \
    --split_id 1 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height \
    --coordinate_frame camera

cd ../..

time python bullet/metrics.py \
    --dataset_dir ~/datasets/ego_v005 \
	--pred_path ~/outputs/ego_v005/test.json

time python bullet/visualize.py \
    --dataset_dir ~/datasets/ego_v005 \
    --pred_path ~/outputs/ego_v005/test.json \
    --output_dir ~/analysis/ego_v005