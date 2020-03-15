cd ns_vqa_dart/scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/stacking_v003 \
    --dataset_dir ~/datasets/stacking_v003 \
    --train_start_id 0 \
    --train_end_id 20000 \
    --eval_start_id 20000 \
    --eval_end_id 22000 \
    --checkpoint_every 2000 \
    --num_iters 60000 \
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
    --run_dir ~/outputs/stacking_v003 \
    --dataset_dir ~/datasets/stacking_v003 \
    --output_path ~/outputs/stacking_v003/pred.json \
    --load_checkpoint_path ~/outputs/stacking_v003/checkpoint_best.pt \
    --eval_start_id 20000 \
    --eval_end_id 22000 \
    --height 480 \
    --width 480 \
    --pred_attr \
    --pred_size \
    --pred_position \
    --pred_up_vector \
    --coordinate_frame camera \
    --num_workers 8

cd ../../..

python ns_vqa_dart/bullet/html_images.py \
    --dataset_dir ~/datasets/stacking_v003 \
	--pred_path ~/outputs/stacking_v003/pred.json \
    --html_dir ~/html/stacking_v003 \
    --coordinate_frame camera \
    --n_objects 100

python ns_vqa_dart/bullet/html.py \
    --html_dir ~/html/stacking_v003 \
    --show_objects
