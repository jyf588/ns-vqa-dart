rm -rf ~/workspace/lucas/unity/Captures/temp
time python demo/run_unity_from_states.py \
    --states_dir ~/data/states/planning_v003_100 \
    --start_id 0 \
    --end_id 100

rm -rf ~/data/planning_v003_100/unity_output
mkdir -p ~/data/planning_v003_100/unity_output
time cp -r ~/workspace/lucas/unity/Captures/temp ~/data/planning_v003_100/unity_output/images
time cp -r ~/data/temp_unity_data ~/data/planning_v003_100/unity_output/json
time zip -r planning_v003_100.zip planning_v003_100
time rsync -azP planning_v003_100.zip sydney:~/mguo/data/datasets/
