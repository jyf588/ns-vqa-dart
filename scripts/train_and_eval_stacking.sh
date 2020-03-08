cd scene_parse/attr_net

time python run_train.py \
    --dataset dash \
    --run_dir ~/outputs/stacking_v001 \
    --dataset_dir ~/datasets/stacking_v001 \
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
    --run_dir ~/outputs/stacking_v001 \
    --dataset_dir ~/datasets/stacking_v001 \
    --output_path ~/outputs/stacking_v001/pred.json \
    --load_checkpoint_path ~/outputs/stacking_v001/checkpoint_best.pt \
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

cd ../..

python bullet/html_images.py \
    --dataset_dir ~/datasets/stacking_v001 \
	--pred_path ~/outputs/stacking_v001/pred.json \
    --html_dir ~/html/stacking_v001 \
    --coordinate_frame camera \
    --n_examples 100

python bullet/html.py \
    --html_dir ~/html_stacking_v001
