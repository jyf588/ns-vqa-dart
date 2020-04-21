cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/stacking_v002_cyl \
    --dataset_dir ~/datasets/stacking_v002_cyl \
    --train_start_id 0 \
    --train_end_id 7000 \
    --eval_start_id 7000 \
    --eval_end_id 8000 \
    --checkpoint_every 2000 \
    --num_iters 100 \
    --height 480 \
    --width 480 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/stacking_v002_cyl \
    --dataset_dir ~/datasets/stacking_v002_cyl \
    --output_path ~/outputs/stacking_v002_cyl/pred.json \
    --load_checkpoint_path ~/outputs/stacking_v002_cyl/checkpoint_best.pt \
    --eval_start_id 7000 \
    --eval_end_id 8000 \
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
    --dataset_dir ~/datasets/stacking_v002_cyl \
	--pred_path ~/outputs/stacking_v002_cyl/pred.json \
    --html_dir ~/html/stacking_v002_cyl \
    --coordinate_frame camera \
    --n_objects 100

python bullet/html.py \
    --html_dir ~/html/stacking_v002_cyl \
    --show_objects
