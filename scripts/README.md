## Requirements

This section contains prerequisites to running the DASH system.

First, clone the main repo and its submodules.
```
git clone https://github.com/jyf588/pytorch-rl-bullet.git
git submodule update --init --recursive
```

Next, create a conda environment.
```
conda create -n dash
conda activate dash
```

Install python packages.
```
cd ns_vqa_dart/scripts
conda install pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

OpenAI baselines:
```
git clone https://github.com/openai/baselines.git
cd baselines
cp <path_to_pytorch_rl_bullet_repo>/baseline_patches/running_mean_std.py baselines/common/
cp <path_to_pytorch_rl_bullet_repo>/baseline_patches/setup.py .
pip install -e .
```

Pytorch installation:
```

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Issue when using the command below: https://github.com/facebookresearch/detectron2/issues/55
# pip install torch==1.4 torchvision==0.5 -f https://download.pytorch.org/whl/cu101/torch_stable.html
```

Check that PyTorch is properly installed:
```
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
print(torch.version.cuda)
```

Make sure that your torch CUDA version printed out matches your CUDA version.

Another command to try:
```
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

Check the gcc version. gcc & g++ ≥ 5 are required.
```
gcc --version
```

Install Detectron2 and pycocotools:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Try importing detectron2:
```
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
```

## Generating Vision Module Datasets

First, here is a changelog of dataset versions and the diffs between successive
versions:

- `planning_v004`, `placing_v004`, `stacking_v004` (May 8, 2020)
- `planning_v003` and `placing_v003` (April 17, 2020)
  - Change the camera "look at" locations:
    - `planning_v003`: Instead of looking at every single object, look once at the center of the distribution of object locations and heights.
    - `placing_v003`: Instead of looking at every single object, look once at the top of the bottom object.
  - FIX: Shadow bias is removed, now shadow and object are joined instead of 
  having a gap between them.
- `dash_v005` (April 14, 2020)
  - Changed from `placing_v002` (100 trials) to `placing_v003` (1500 trials).
  - Changed from 50/50 split between planning and placing to 25/75 split.
- `dash_v004` (Apr 13, 2020)
  - Update to use Yifeng's new placing policy: `0404_0_n_20_40`.
  - Changed from `placing.sh` to `placing_v002.sh`.
- `dash_v003`: 
  - FIX: Update the transformation of y labels (specifically, position and 
  orientation of objects) from the incorrect bworld -> ucam to the correct 
  bworld -> bshoulder -> ushoulder -> ucam transformation.
  - FIX: Images; Unity delayed rendering vs. snapshot where arm was never 
    really holding any objects before.
  - FIX: Bug where saved camera orientation is changed from xyyw -> xyzw.
- `dash_v002`: The first dataset to train both planning and placing, together.

Step 1. Generate policy rollout states for placing and stacking. 
(ETA: 1 hr 20 mins)

```
./ns_vqa_dart/scripts/01_generate_policy_rollouts.sh
```

Step 2. Generate states for planning, placing, and stacking. 
(ETA: 40 minutes)

```
time ./ns_vqa_dart/scripts/planning/v003_20K/01_generate_states.sh
time ./ns_vqa_dart/scripts/placing/v003_2K_20K/01_generate_states.sh
time ./ns_vqa_dart/scripts/stacking/v003_2K_20K/01_generate_states.sh
```

Step 2. On the Unity machine, run the following to transfer the states to the
Unity machine.
(ETA: 1 minute 22 seconds)

```
time ./ns_vqa_dart/scripts/planning_v003_20K/02_transfer_states.sh
time ./ns_vqa_dart/scripts/placing_v003_2K_100/02_transfer_states.sh
```

Step 3. Generate Unity images from the states, and transfer the images to the 
machine where training will occur. 

(ETA: 1 hour 30 minutes for generation, 3 hours for transfer)

Note: Currently you need to update the `end_id` of the `DatasetLoader` in the
script.

```
time ./ns_vqa_dart/scripts/planning_v003_20K/03_generate_unity_images.sh
time ./ns_vqa_dart/scripts/placing_v003_2K_100/03_generate_unity_images.sh
```

Step 4. Generate the dataset and run training and evaluation.

ETA: 30 minutes for generation, 3 hours for training
WARNING: Make sure sydney has enough space before running this! Roughly 120 GB
of space is needed.


```
time ./ns_vqa_dart/scripts/planning_v003_20K/04_generate_and_run.sh
./ns_vqa_dart/scripts/placing_v003_2K_100/04_generate_and_run.sh
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

First time running the scripts will take a few minutes. (Usually when yellow
output shows up)

April 15, 2020: Re-ran `sudo docker build -t openrave-ha:v3 .` with 
the following command (because `inmoov.git` was updated):
```
sudo docker build --no-cache -t openrave-ha:v3 .
```
And then restarted both containers.

April 28, 2020: Trying to re-add reaching and retracting.
1. Pull the or_planning_scripts repo.
2. Stop running existing scripts.
3. Start a third container.
4. Run each of the following commands in the three containers:
    1. `python move_single 0`  # Reach
    2. `python move_single 1`  # Place
    3. `python move_single 2 l`  # Retract

## Detectron


### Run a pre-trained detectron2 model

First, download a random image from the COCO dataset:

```
cd ns_vqa_dart/scene_parse/detectron2
wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
```

Run the pretrained model:
```
python pretrained.py
```

The resulting image is written to `detectron2/result_im.png`.

### Train on balloon

Get the dataset:
```
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
unzip balloon_dataset.zip > /dev/null
```

```
python dash.py
```
