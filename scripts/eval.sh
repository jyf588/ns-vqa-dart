cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v008 \
    --dataset_dir ~/datasets/ego_v008 \
    --output_path ~/outputs/ego_v008/test.json \
    --load_checkpoint_path ~/outputs/ego_v008/checkpoint_best.pt \
    --split_id 20000 \
    --height 480 \
    --width 480 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera

cd ../..

time python bullet/metrics.py \
    --dataset_dir ~/datasets/ego_v008 \
	--pred_path ~/outputs/ego_v008/test.json \
    --coordinate_frame camera

python html/html_images.py \
    --dataset_dir ~/datasets/ego_v008 \
	--pred_path ~/outputs/ego_v008/test.json \
    --output_dir ~/analysis/ego_v008 \
    --coordinate_frame camera \
    --n_examples 200

python html/html.py \
    --dataset ego_v007 \
    --start_img_id 20000 \
    --end_img_id 20049
