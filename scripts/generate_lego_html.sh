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
