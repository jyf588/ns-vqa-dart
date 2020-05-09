Installation instructions: See INSTALL.md.

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

Step 1. Generate states for planning, placing, and stacking. 
(ETA: TODO)

```
time ./ns_vqa_dart/scripts/01_generate_states.sh
```

Step 3. Generate Unity images from the states.
(ETA: TODO)

```
time ./ns_vqa_dart/scripts/02_generate_unity_images.sh
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
