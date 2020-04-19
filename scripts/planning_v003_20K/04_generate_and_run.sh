cd ~/mguo/data/datasets
mkdir planning_v003_20K
mv planning_v003_20K.zip planning_v003_20K/
cd planning_v003_20K
time unzip planning_v003_20K.zip
rm -rf unity_output
mv planning_v003_20K/unity_output .
rm -rf planning_v003_20K
cd ~/workspace/pytorch-rl-bullet

time ./ns_vqa_dart/scripts/planning_v003_20K/generate.sh
time ./ns_vqa_dart/scripts/planning_v003_20K/dry_run.sh
time ./ns_vqa_dart/scripts/planning_v003_20K/run.sh
