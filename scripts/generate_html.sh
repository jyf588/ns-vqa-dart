python html/html_images.py \
    --dataset_dir ~/datasets/ego_v006 \
	--pred_path ~/outputs/ego_v006/test.json \
    --output_dir ~/analysis/ego_v006 \
    --coordinate_frame camera \
    --n_examples 10

python html/html.py \
    --dataset ego_v006 \
    --start_img_id 20000 \
    --end_img_id 20002