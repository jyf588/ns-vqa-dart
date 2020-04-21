cd ~/mguo/data/datasets
mkdir stacking_v003_2K_100
mv stacking_v003_2K_100.zip stacking_v003_2K_100/
cd stacking_v003_2K_100
time unzip stacking_v003_2K_100.zip
rm -rf unity_output
cd stacking_v003_2K_100
mv unity_output ../
cd ..
rm -rf stacking_v003_2K_100
cd ~/workspace/pytorch-rl-bullet

time ./ns_vqa_dart/scripts/stacking_v003_2K_100/generate.sh
time ./ns_vqa_dart/scripts/stacking_v003_2K_100/dry_run.sh
time ./ns_vqa_dart/scripts/stacking_v003_2K_100/run.sh
