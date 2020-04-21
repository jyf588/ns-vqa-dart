cd ~/mguo/data/datasets
mkdir placing_v003_10_100
mv placing_v003_10_100.zip placing_v003_10_100/
cd placing_v003_10_100
time unzip placing_v003_10_100.zip
rm -rf unity_output
cd placing_v003_10_100
mv unity_output ../
cd ..
rm -rf placing_v003_10_100
cd ~/workspace/pytorch-rl-bullet

time ./ns_vqa_dart/scripts/placing_v003_10_100/generate.sh
time ./ns_vqa_dart/scripts/placing_v003_10_100/dry_run.sh
time ./ns_vqa_dart/scripts/placing_v003_10_100/run.sh
