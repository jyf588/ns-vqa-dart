ROOT_DIR=/media/sdc3/mguo

cd $ROOT_DIR/data/datasets
mkdir placing_v003_2K_20K
mv placing_v003_2K_20K.zip placing_v003_2K_20K/
cd placing_v003_2K_20K
time unzip placing_v003_2K_20K.zip
rm -rf unity_output
cd placing_v003_2K_20K
mv unity_output ../
cd ..
rm -rf placing_v003_2K_20K
cd ~/workspace/pytorch-rl-bullet

time ./ns_vqa_dart/scripts/placing/v003_2K_20K/generate.sh
time ./ns_vqa_dart/scripts/placing/v003_2K_20K/dry_run.sh
time ./ns_vqa_dart/scripts/placing/v003_2K_20K/run.sh
