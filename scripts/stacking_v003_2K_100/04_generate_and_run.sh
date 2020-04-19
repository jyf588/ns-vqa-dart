cd ~/mguo/data/datasets
mkdir planning_v003_100
mv planning_v003_100.zip planning_v003_100/
cd planning_v003_100
time unzip planning_v003_100.zip
rm -rf unity_output
cd planning_v003_100
mv unity_output ../
cd ..
rm -rf planning_v003_100
cd ~/workspace/pytorch-rl-bullet

time ./ns_vqa_dart/scripts/planning_v003_100/generate.sh
time ./ns_vqa_dart/scripts/planning_v003_100/dry_run.sh
time ./ns_vqa_dart/scripts/planning_v003_100/run.sh
