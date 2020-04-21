cd ~/mguo/data/datasets
mkdir stacking_v003_2K_20K
mv stacking_v003_2K_20K.zip stacking_v003_2K_20K/
cd stacking_v003_2K_20K
time unzip stacking_v003_2K_20K.zip
rm -rf unity_output
cd stacking_v003_2K_20K
mv unity_output ../
cd ..
rm -rf stacking_v003_2K_20K
cd ~/workspace/pytorch-rl-bullet

time ./ns_vqa_dart/scripts/stacking_v003_2K_20K/generate.sh
time ./ns_vqa_dart/scripts/stacking_v003_2K_20K/dry_run.sh
time ./ns_vqa_dart/scripts/stacking_v003_2K_20K/run.sh
