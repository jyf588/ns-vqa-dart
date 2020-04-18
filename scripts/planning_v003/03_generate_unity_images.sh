rm -rf ~/workspace/lucas/unity/Captures/temp
time python demo/run_unity_from_states.py \
    --states_dir ~/data/states/dash_v006

rm -rf ~/data/dash_v006/unity_output
mkdir -p ~/data/dash_v006/unity_output
time cp -r ~/workspace/lucas/unity/Captures/temp ~/data/dash_v006/unity_output/images
time cp -r ~/data/temp_unity_data ~/data/dash_v006/unity_output/json
time zip -r dash_v006.zip dash_v006
time rsync -azP dash_v006.zip sydney:~/mguo/data/datasets/dash_v006/
