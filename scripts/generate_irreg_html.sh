python html/html_images.py \
    --dataset_dir ~/datasets/cup \
	--pred_path ~/outputs/cup/test.json \
    --output_dir ~/analysis/cup \
    --coordinate_frame camera \
    --n_examples 30

python html/html.py \
    --dataset cup \
    --start_img_id 0 \
    --end_img_id 30
