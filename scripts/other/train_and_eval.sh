cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v008_224 \
    --dataset_dir ~/datasets/ego_v008 \
    --split_id 20000 \
    --checkpoint_every 2000 \
    --num_iters 60000 \
    --height 224 \
    --width 224 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v008_224 \
    --dataset_dir ~/datasets/ego_v008 \
    --output_path ~/outputs/ego_v008_224/test.json \
    --load_checkpoint_path ~/outputs/ego_v008_224/checkpoint_best.pt \
    --split_id 20000 \
    --height 224 \
    --width 224 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

cd ../..

python html/html_images.py \
    --dataset_dir ~/datasets/ego_v008 \
	--pred_path ~/outputs/ego_v008_224/test.json \
    --output_dir ~/analysis/ego_v008_224 \
    --coordinate_frame camera \
    --n_examples 100

python html/html.py \
    --dataset_dir datasets/ego_v008 \
    --analysis_dir analysis/ego_v008_224 \
    --start_img_id 20000 \
    --end_img_id 20025
