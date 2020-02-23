cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/cup \
    --dataset_dir ~/datasets/cup \
    --output_path ~/outputs/cup/test.json \
    --load_checkpoint_path ~/outputs/ego_v008/checkpoint_best.pt \
    --split_id 0 \
    --height 480 \
    --width 480 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

cd ../..

python html/html_images.py \
    --dataset_dir ~/datasets/cup \
	--pred_path ~/outputs/cup/test.json \
    --output_dir ~/analysis/cup \
    --coordinate_frame camera \
    --n_examples 10

python html/html.py \
    --dataset_dir datasets/cup \
    --analysis_dir analysis/cup \
    --start_img_id 0 \
    --end_img_id 10
