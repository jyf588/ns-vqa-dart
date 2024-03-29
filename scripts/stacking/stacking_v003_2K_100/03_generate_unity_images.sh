rm -rf ~/workspace/lucas/unity/Captures/temp
rm -rf ~/data/temp_unity_data
time python system/run_unity_from_states.py \
    --states_dir ~/data/states/stacking_v003_2K_100 \
    --start_id 0 \
    --end_id 100 \
    --camera_control stack \
    --out_dir /Users/michelleguo/data/temp_unity_data

rm -rf ~/data/stacking_v003_2K_100/unity_output
mkdir -p ~/data/stacking_v003_2K_100/unity_output
time cp -r ~/workspace/lucas/unity/Captures/temp ~/data/stacking_v003_2K_100/unity_output/images
time cp -r ~/data/temp_unity_data ~/data/stacking_v003_2K_100/unity_output/json
cd ~/data
rm stacking_v003_2K_100.zip
time zip -r stacking_v003_2K_100.zip stacking_v003_2K_100
time rsync -azP stacking_v003_2K_100.zip sydney:~/mguo/data/datasets/
