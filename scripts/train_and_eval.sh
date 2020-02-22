cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v007 \
    --dataset_dir ~/datasets/ego_v007 \
    --split_id 20000 \
    --checkpoint_every 2000 \
    --num_iters 60000 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v007 \
    --dataset_dir ~/datasets/ego_v007 \
    --output_path ~/outputs/ego_v007/test.json \
    --load_checkpoint_path ~/outputs/ego_v007/checkpoint_best.pt \
    --split_id 20000 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

cd ../..

time python bullet/metrics.py \
    --dataset_dir ~/datasets/ego_v007 \
	--pred_path ~/outputs/ego_v007/test.json \
    --coordinate_frame camera

python html/html_images.py \
    --dataset_dir ~/datasets/ego_v007 \
	--pred_path ~/outputs/ego_v007/test.json \
    --output_dir ~/analysis/ego_v007 \
    --coordinate_frame camera \
    --n_examples 100

python html/html.py \
    --dataset ego_v007 \
    --start_img_id 0 \
    --end_img_id 100
