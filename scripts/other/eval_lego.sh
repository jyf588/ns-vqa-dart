cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/lego \
    --dataset_dir ~/datasets/lego \
    --output_path ~/outputs/lego/test.json \
    --load_checkpoint_path ~/outputs/ego_v006/checkpoint_best.pt \
    --split_id 0 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera

cd ../..

time python bullet/metrics.py \
    --dataset_dir ~/datasets/lego \
	--pred_path ~/outputs/lego/test.json \
    --coordinate_frame camera

python html/html_images.py \
    --dataset_dir ~/datasets/lego \
	--pred_path ~/outputs/lego/test.json \
    --output_dir ~/analysis/lego \
    --coordinate_frame camera \
    --n_examples 30

python html/html.py \
    --dataset lego \
    --start_img_id 0 \
    --end_img_id 30
