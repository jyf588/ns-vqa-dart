cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/lego \
    --dataset_dir ~/datasets/lego \
    --output_path ~/outputs/lego/test.json \
    --load_checkpoint_path ~/outputs/ego_v005/checkpoint_best.pt \
    --split_id 0 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height \
    --coordinate_frame camera

cd ../..

time python bullet/metrics.py \
    --dataset_dir ~/datasets/lego \
	--pred_path ~/outputs/lego/test.json \
    --coordinate_frame camera
