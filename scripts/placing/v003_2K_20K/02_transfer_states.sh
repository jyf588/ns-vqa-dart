ROOT_DIR=/media/sdc3/mguo

time rsync -azP sydney:$ROOT_DIR/data/states/full/placing_v003_10_100.zip ~/data/states/
cd ~/data/states
rm -rf placing_v003_10_100
time unzip placing_v003_10_100.zip
