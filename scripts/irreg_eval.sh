cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/cup \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/cup \
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
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/cup \
	--pred_path ~/outputs/cup/test.json \
    --html_dir ~/analysis/cup \
    --coordinate_frame camera \
    --n_examples 10

python html/html.py \
    --html_dir ~/analysis/cup
    