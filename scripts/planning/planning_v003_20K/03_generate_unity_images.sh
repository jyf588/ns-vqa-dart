rm -rf ~/workspace/lucas/unity/Captures/temp
time python demo/run_unity_from_states.py \
    --states_dir ~/data/states/planning_v003_20K \
    --start_id 0 \
    --end_id 20000 \
    --camera_control center \
    --out_dir /Users/michelleguo/data/temp_unity_data

rm -rf ~/data/planning_v003_20K/unity_output
mkdir -p ~/data/planning_v003_20K/unity_output
time cp -r ~/workspace/lucas/unity/Captures/temp ~/data/planning_v003_20K/unity_output/images
time cp -r ~/data/temp_unity_data ~/data/planning_v003_20K/unity_output/json
cd ~/data
time zip -r planning_v003_20K.zip planning_v003_20K
time rsync -azP planning_v003_20K.zip sydney:~/mguo/data/datasets/
