python bullet/generate.py \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/irreg \
    --n_examples 2

cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/irreg \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/irreg \
    --output_path ~/outputs/irreg/test.json \
    --load_checkpoint_path ~/outputs/ego_v008/checkpoint_best.pt \
    --start_id 0 \
    --end_id 1 \
    --height 480 \
    --width 480 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

cd ../..

python bullet/html_images.py \
    --dataset_dir /media/michelle/68B62784B62751BC/datasets/irreg \
	--pred_path ~/outputs/irreg/test.json \
    --html_dir ~/html/irreg \
    --coordinate_frame camera

python bullet/html.py \
    --html_dir ~/html/irreg \
    --show_objects
