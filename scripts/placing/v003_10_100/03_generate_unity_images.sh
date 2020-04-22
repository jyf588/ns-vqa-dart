ROOT_DIR=/media/sdc3/mguo

rm -rf ~/workspace/lucas/unity/Captures/temp
rm -rf ~/data/temp_unity_data
time python system/run_unity_from_states.py \
    --states_dir ~/data/states/placing_v003_10_100 \
    --start_id 0 \
    --end_id 100 \
    --camera_control stack \
    --out_dir /Users/michelleguo/data/temp_unity_data

rm -rf ~/data/placing_v003_10_100/unity_output
mkdir -p ~/data/placing_v003_10_100/unity_output
time cp -r ~/workspace/lucas/unity/Captures/temp ~/data/placing_v003_10_100/unity_output/images
time cp -r ~/data/temp_unity_data ~/data/placing_v003_10_100/unity_output/json
cd ~/data
rm placing_v003_10_100.zip
time zip -r placing_v003_10_100.zip placing_v003_10_100
time rsync -azP placing_v003_10_100.zip sydney:$ROOT_DIR/data/datasets/
