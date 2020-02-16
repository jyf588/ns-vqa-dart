cd scene_parse/attr_net

time python run_test.py \
    --dataset dash \
    --run_dir ~/outputs/ego_001 \
    --dataset_dir ~/datasets/ego_001 \
    --output_path ~/outputs/ego_001/test.json \
    --split_id 0 \
    --pred_attr \
    --pred_position \
    --pred_up_vector \
    --pred_height

cd /home/michelle/workspace/bullet-vision-inference

time python compute_metrics.py \
	--true_path ~/datasets/large_cyl/annotations.json \
	--pred_path ~/outputs/ego_001/test.json \
    --eval_position \
    --eval_z_dir \
    --eval_z_size \
	--split_id 20000 \
	--n_examples 22000
