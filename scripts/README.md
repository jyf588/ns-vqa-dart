## Generating vision module datasets

First, here is a changelog of dataset versions and the diffs between successive
versions:

- `dash_v002`: The first dataset to train both planning and placing, together.
- `dash_v003`: 
  - FIX: Update the transformation of y labels (specifically, position and 
  orientation of objects) from the incorrect bworld -> ucam to the correct 
  bworld -> bshoulder -> ushoulder -> ucam transformation.
  - FIX: Images; Unity delayed rendering vs. snapshot where arm was never 
    really holding any objects before.
  - FIX: Bug where saved camera orientation is changed from xyyw -> xyzw.
- `dash_v004` (Apr 13, 2020)
  - Update to use Yifeng's new placing policy: `0404_0_n_20_40`.
  - Changed from `placing.sh` to `placing_v002.sh`.
- `dash_v005` (April 14, 2020)
  - Changed from `placing_v002` (100 trials) to `placing_v003` (1500 trials).
  - Changed from 50/50 split between planning and placing to 25/75 split.

Step 1. Generate states for planning and placing.

```
# First, generate states for planning and stacking.

time ./ns_vqa_dart/scripts/states/planning_v003.sh  (ETA: 10 seconds)
time ./ns_vqa_dart/scripts/states/placing_v003.sh (ETA: 38 minutes)

time ./ns_vqa_dart/scripts/states/process_placing_v003.sh (ETA: 3 minutes)
time ./ns_vqa_dart/scripts/dash_v005_20K/combine.sh (ETA 4 seconds)
```

Step 2. Zip, transfer, and unzip the states to the machine where you will be
running Unity. Note: Run this on the unity machine.

```
cd ~/mguo/data/states/full
time zip -r dash_v005_20K.zip dash_v005_20K (1 second)
time ./ns_vqa_dart/scripts/dash_v005_20K/transfer_states.sh (1 minute 22 seconds)
```

Step 3. Generate Unity images from the states. (ETA: 1 hour 30 minutes)
Note: Currently you need to update the `end_id` of the `DatasetLoader` in the
script.
```
rm -rf ~/workspace/lucas/unity/Captures/temp
time python demo/run_unity_from_states.py --states_dir ~/data/states/dash_v005_20K
```

Step 4. Zip up and scp the generated Unity data to the machine where 
training will occur.

```
# If the directory already exists:
rm -rf ~/data/dash_v005_20K/unity_output
mkdir -p ~/data/dash_v005_20K/unity_output
time cp -r ~/workspace/lucas/unity/Captures/temp ~/data/dash_v005_20K/unity_output/images
time cp -r ~/data/temp_unity_data ~/data/dash_v005_20K/unity_output/json

# Zip up the data (ETA: 15 minutes)
cd ~/data
time zip -r dash_v005_20K.zip dash_v005_20K

# Transfer the the data
# ETA: 
#   100: 1 minute 30 seconds
#   20K: 1 hour 45 minutes
# WARNING: Make sure there is enough space in sydney!
time rsync -azP dash_v005_20K.zip sydney:~/mguo/data/datasets/dash_v005_20K/
```

Unzip the data.
ETA:
- 100: 1 second
- 20K: 1 minute 12 seconds
WARNING: Make sure there is enough space in sydney!
```
cd ~/mguo/data/datasets/dash_v005_20K
time unzip dash_v005_20K.zip
cd dash_v005_20K
mv unity_output ../
cd ..
du -sh ./dash_v005_20K  # Make sure this folder is empty
rm -rf dash_v005_20K
rm dash_v005_20K.zip
```

Step 5. Generate the dataset for training and testing.
ETA: 
- 100: 7 seconds
- 20K: 33 minutes

```
# WARNING: Make sure to clear up space before running this!
time ./ns_vqa_dart/scripts/dash_v005_20K/generate.sh
```

Step 6. (Optional) Check whether there are any corrupt pickle files.

```
./ns_vqa_dart/scripts/dash_v005_20K/check_pickles.sh
```

## Training and testing the vision module on datasets

To run training and testing on a tiny subset of the dataset for a few 
iterations as a dry run:

```
time ./ns_vqa_dart/scripts/dash_v005_20K/dry_run.sh
```

To run training and testing on the full dataset:

```
time ./ns_vqa_dart/scripts/dash_v005_20K/run.sh
```

## To visualize results in an HTML webpage

Start the server:
```
cd /media/michelle/68B62784B62751BC/html
python -m http.server
```

## Generating result tables

### Table 1: Results on the full system

Below are instructions on how to generate a demo video of Lucas.

Step 1. First, follow instructions below on running reaching and transporting
OpenRAVE programs in docker containers.

Step 2. Run the following bash script to generate the demos.
```
./scripts/table1/gv5_pv9_gt_delay.sh
```

Step 3. Transfer the poses to the machine where Unity will be run.
```
rsync -azP ~/demo_poses ~/workspace/lucas/
```

### Table 2: Results on stacking

```
# Ground truth
./scripts/table2/delay_box.sh
./scripts/table2/delay_cyl.sh

# Vision module
./scripts/table2/delay_vision_box.sh
./scripts/table2/delay_vision_cyl.sh

# Baseline
./scripts/table2/baseline_box.sh
./scripts/table2/baseline_cyl.sh
```

## NLP-related installation instructions

Mainly to run `main_sim_stack_new.py`:

```
pip install spacy nltk
python -m spacy download en_core_web_sm
```

## Instructions for Building & Running OpenRAVE:

First, build the image.

```
git clone git@github.com:jyf588/openrave-installation.git
cd openrave-installation
git checkout my_docker
cd nvidia-docker
sudo docker build -t openrave-ha:v0 .
cd ../openrave-docker
sudo docker build -t openrave-ha:v2 .
cd ../or-my-docker
sudo docker build -t openrave-ha:v3 .
```

Next, run the container of the built image:
(create a container with access to data from the host machine create a folder "container_data" in the home directory)

```
xhost +si:localuser:root
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/container_data:/data --name openravecont openrave-ha:v3 /bin/bash
```

Troubleshooting
```
>>> docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
Solution: https://devtalk.nvidia.com/default/topic/1061452/docker-and-nvidia-docker/could-not-select-device-driver-quot-quot-with-capabilities-gpu-/

Inside the running container, see if you can use Firefox and openrave with GUI.
```
source bashrc
glxgears
firefox
```

To run openrave examples: (https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(openrave-config --python-dir)/openravepy/_openravepy_
export PYTHONPATH=$PYTHONPATH:$(openrave-config --python-dir)
openrave.py --example hanoi
```

Other useful docker commands (with sudo): 

```
docker ps --all (list containers)
docker rm -f contrainer_number (kill and remove container)
docker image list
docker image rm image_number
docker cp <src_path> openravecont:<dst_path>
```

To run OpenRAVE for DASH:

Outside of docker container:
```
cd ~/container_data
git clone https://github.com/jyf588/or_planning_scripts
```

Update `tabletop_2.kinbody.xml` to the following:

```
<KinBody name="box0">
  <Body name="base">
    <Geom type="box">
      <extents>1.3 0.6 0.05</extents>
      <translation>0.2 0.1 -0.13</translation>
      <diffusecolor>0.6 0 0</diffusecolor>
    </Geom>
  </Body>
</KinBody>
```

Inside of docker container, run the following to run reaching:
```
source bashrc
cd /data/or_planning_scripts
python move_single.py 0
```

Start a second docker container session to run transport:
If you need to run reach_single.py also, open another session for the same 
container:
```
sudo docker exec -it openravecont /bin/bash
source bashrc
cd /data/or_planning_scripts
python move_single.py 1
```
