cd ~/mguo/data/datasets/dash_v006
time unzip dash_v006.zip
cd dash_v006
mv unity_output ../
cd ..
rm -rf dash_v006
rm dash_v006.zip
time ./ns_vqa_dart/scripts/dash_v006/generate.sh
time ./ns_vqa_dart/scripts/dash_v006/dry_run.sh
time ./ns_vqa_dart/scripts/dash_v006/run.sh
