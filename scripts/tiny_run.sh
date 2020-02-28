python bullet/generate.py \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/cup \
    --n_examples 20

cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v007 \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/ego_v008 \
    --split_id 21990 \
    --checkpoint_every 2000 \
    --num_iters 20 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_v007 \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/ego_v008 \
    --output_path ~/outputs/ego_v007/test.json \
    --load_checkpoint_path ~/outputs/ego_v007/checkpoint_best.pt \
    --split_id 21990 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

cd ../..

python html/html_images.py \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/ego_v008 \
	--pred_path ~/outputs/ego_v007/test.json \
    --output_dir ~/analysis/ego_v007 \
    --coordinate_frame camera \
    --n_examples 10

python html/html.py \
    --dataset ego_v007 \
    --start_img_id 21990 \
    --end_img_id 21992
