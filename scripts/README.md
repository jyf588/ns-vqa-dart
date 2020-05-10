# DASH

Installation instructions: See INSTALL.md.

## System

First, start the python server:

```
python system/run.py
```

Next, run the Unity executable:
```
./LinuxBuildLocalhost.x86_64
```

## Tables and Figures

### Table 1

First, generate the test scenes:

```
time python scenes/generate.py table1  # ETA: 1 second
```

### Table 2
TODO

### Table 3
TODO

## Vision Module

### Vision Datasets

To generate your own datasets for training the vision module, run the following
commands. You can choose to either run the tiny dataset or the full dataset.

Step 1. Generate scenes for planning, placing, and stacking. 

```
time python scenes/generate.py vision  # ETA: 5 seconds
time python scenes/generate.py vision_tiny  # ETA: 1 second
```

Step 2. Generate Unity images from the states.
(ETA: TODO)

```
time ./ns_vqa_dart/scripts/02_generate_unity_images.sh
```

### Vision Training

You can train your own segmentation module and vision module.

Train a segmentation module:

```
TODO
```

Train a vision module:

```
TODO
```

### Vision dataset versions 
Here is a changelog of dataset versions and the diffs between successive
versions:

- `planning_v004`, `placing_v004`, `stacking_v004` (May 8, 2020)
  - Use system code instead of enjoy.py to generate policy rollouts.
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
