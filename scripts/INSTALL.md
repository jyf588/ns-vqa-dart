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

## Instructions for Building & Running OpenRAVE:

### Prerequisites
1. Docker (https://docs.docker.com/engine/install/ubuntu/)
2. `nvidia-container-toolkit` (https://github.com/NVIDIA/nvidia-docker#ubuntu-16041804-debian-jessiestretchbuster)

### Building docker images

First, build the images. This can take a few minutes.

```
git clone git@github.com:jyf588/openrave-installation.git
cd openrave-installation
git checkout my_docker

cd nvidia-docker
time sudo docker build -t openrave-ha:v0 .

cd ../openrave-docker
time sudo docker build -t openrave-ha:v2 .

cd ../or-my-docker
time sudo docker build -t openrave-ha:v3 .  # (ETA: 30 seconds)
```

## Running OpenRAVE

On the host machine, create a folder which we will grant container access to.
```
mkdir <path_to_container_data>/container_data
```

On the host machine, clone the repository containing OpenRAVE scripts we will run into the `container_data` folder we just created.
```
cd <path_to_container_data>/container_data
git clone git@github.com:jyf588/or_planning_scripts.git
```

Start three separate docker containers, each in separate terminal sessions, with the following commands:
```
xhost +si:localuser:root

# Start the first container
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v <path_to_container_data>/container_data:/data --name openravecont openrave-ha:v3 /bin/bash

# Run this twice in separate terminal sessions, for each of the second and third containers
sudo docker exec -it openravecont /bin/bash
```

Then, in each of the three containers, run the following commands:
```
source bashrc
cd /data/or_planning_scripts
```

Finally, run the following commands in each of the three containers:
```
python move_single.py 0
python move_single.py 1
python move_single.py 2 l
```
These correspond to reach, move and retract, respectively.

If you need to rebuild an image, run the following command (replace `v3` with the desired tag).
```
sudo docker build --no-cache -t openrave-ha:v3 .
```
And make sure to restart the containers after rebuilding.


Troubleshooting
```
>>> docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
Solution: https://devtalk.nvidia.com/default/topic/1061452/docker-and-nvidia-docker/could-not-select-device-driver-quot-quot-with-capabilities-gpu-/


Inside the running container, see if you can use Firefox and openrave with GUI.
```
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

Quick commands to spin up multiple OpenRAVE's:

### Container v1:
```
xhost +si:localuser:root
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/container_data_v1:/data --name openravecont_v1 openrave-ha:v3 /bin/bash
sudo docker exec -it openravecont_v1 /bin/bash
sudo docker exec -it openravecont_v1 /bin/bash
```

### Reach
```
source bashrc
cd /data/or_planning_scripts
python move_single.py 0
```

### Move
```
source bashrc
cd /data/or_planning_scripts
python move_single.py 1
```

### Retract
```
source bashrc
cd /data/or_planning_scripts
python move_single.py 2 l
```

### Container v2:
````
xhost +si:localuser:root
sudo docker run --gpus=all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ~/container_data_v2:/data --name openravecont_v2 openrave-ha:v3 /bin/bash
source bashrc
cd /data/or_planning_scripts
python move_single.py 0
```

```
sudo docker exec -it openravecont_v2 /bin/bash
source bashrc
cd /data/or_planning_scripts
python move_single.py 1
```

```
sudo docker exec -it openravecont_v2 /bin/bash
source bashrc
cd /data/or_planning_scripts
python move_single.py 2 l
```