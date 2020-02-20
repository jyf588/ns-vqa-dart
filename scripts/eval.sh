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
    --coordinate_frame camera

cd ../..

time python bullet/metrics.py \
    --dataset_dir ~/datasets/ego_v006 \
	--pred_path ~/outputs/ego_v006/test.json \
    --coordinate_frame camera

python html/html_images.py \
    --dataset_dir ~/datasets/ego_v006 \
	--pred_path ~/outputs/ego_v006/test.json \
    --output_dir ~/analysis/ego_v006 \
    --coordinate_frame camera \
    --n_examples 200

python html/html.py \
    --dataset ego_v006 \
    --start_img_id 20000 \
    --end_img_id 20049
