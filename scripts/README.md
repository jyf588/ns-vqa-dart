## Generating vision module datasets

Step 1. Generate states for planning and placing.

```
# First, generate states for planning and stacking.
./ns_vqa_dart/scripts/states/planning_v003.sh  (ETA: 10 seconds)
./ns_vqa_dart/scripts/states/placing.sh (ETA: 3 minutes)

# Process and combine states.
./ns_vqa_dart/scripts/states/process_and_combine.sh (ETA 4 seconds)
```

Step 2. Transfer the states (23 M) to the machine where you will be
running Unity.

```
# Transfer the states.
time rsync -azP sydney:~/mguo/data/states/full/dash_v002_5K.zip ~/workspace/lucas/states/  # ETA: 1 second

# Unzip the states.
unzip ~/workspace/lucas/states/dash_v002_5K.zip  # ETA: 1 second
```

Step 3. Generate Unity images from the states.

```
time python scripts/server.py (ETA: 18 minutes)
```

Step 4. Zip up and scp the generated Unity data to the machine where 
training will occur.

```
# Zip up the data (ETA: 2 minutes)
time zip -r dash_v002_100.zip dash_v002_100

# Transfer the the data (ETA: 36 minutes)
time rsync -azP dash_v002_5K.zip sydney:~/mguo/data/datasets/dash_v002_5K/

# Unzip the data. (ETA: 1 second)
time unzip dash_v002_100.zip
```

Step 5. Generate the dataset for training and testing. (ETA: 7 hours)

```
# To generate the full set (ETA: 10 minutes):
./ns_vqa_dart/scripts/dash_v002/generate.sh
```

Step 6. (Optional) Check whether there are any corrupt pickle files.

```
./ns_vqa_dart/scripts/dash_v001/check_pickles.sh
```

## Training and testing the vision module on datasets

To run training and testing on a tiny subset of the dataset for a few 
iterations as a dry run:

```
time ./ns_vqa_dart/scripts/dash_v002/dry_run.sh
```

To run training and testing on the full dataset:

```
./ns_vqa_dart/scripts/dash_v002/run.sh
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
