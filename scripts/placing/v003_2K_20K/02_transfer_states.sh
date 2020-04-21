ROOT_DIR=/media/sdc3/mguo

time rsync -azP sydney:$ROOT_DIR/data/states/full/placing_v003_2K_20K.zip ~/data/states/
cd $ROOT_DIR/data/states
rm -rf placing_v003_2K_20K
time unzip placing_v003_2K_20K.zip
