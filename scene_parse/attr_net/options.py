import argparse
import os
import torch
from typing import *


from ns_vqa_dart.bullet import util


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            "--run_dir",
            default="scratch/test_run",
            type=str,
            help="experiment directory",
        )
        self.parser.add_argument("--dataset", default="dash", type=str, help="dataset")
        self.parser.add_argument(
            "--data_dir", required=True, type=str, help="The dataset directory."
        )
        self.parser.add_argument(
            "--load_checkpoint_path",
            default=None,
            type=str,
            help="load checkpoint path",
        )
        self.parser.add_argument(
            "--checkpoint_t",
            default=None,
            type=int,
            help="Iteration of the loaded checkpoint",
        )
        self.parser.add_argument(
            "--gpu_ids", default="0", type=str, help="ids of gpu to be used"
        )
        self.parser.add_argument(
            "--inference_only",
            action="store_true",
            help="Whether to only run inference.",
        )
        # self.parser.add_argument("--cam_dir", type=str, help="The camera directory.")
        self.parser.add_argument(
            "--height", type=int, default=480, help="The image height."
        )
        self.parser.add_argument(
            "--width", type=int, default=480, help="The image width"
        )
        # self.parser.add_argument('--clevr_mini_img_dir', default='../../data/raw/CLEVR_mini/images', type=str, help='clevr-mini image directory')
        # self.parser.add_argument('--clevr_mini_ann_path', default='../../data/attr_net/objects/clevr_mini_objs.json', type=str, help='clevr-mini objects annotation file')

        self.parser.add_argument(
            "--concat_img",
            default=1,
            type=int,
            help="concatenate original image when sent to network",
        )
        self.parser.add_argument(
            "--with_depth", default=0, type=int, help="include depth info (rgbd)",
        )
        # self.parser.add_argument(
        #     "--camera_control",
        #     type=str,
        #     choices=["all", "center", "stack"],
        #     help="The method of controlling the camera.",
        # )
        # self.parser.add_argument(
        #     "--coordinate_frame",
        #     choices=["world", "camera", "unity_camera"],
        #     help="The coordinate frame to train on.",
        # )
        self.parser.add_argument(
            "--plot_path", type=str, help="The path to save the output plot to.",
        )
        self.parser.add_argument(
            "--fp16", action="store_true", help="Whether to use FP 16."
        )
        self.parser.add_argument(
            "--batch_size", default=20, type=int, help="batch size"
        )
        self.parser.add_argument(
            "--num_workers", default=4, type=int, help="number of workers for loading",
        )
        self.parser.add_argument(
            "--learning_rate", default=0.002, type=float, help="learning rate"
        )

        self.initialized = True

    def parse(
        self,
        opt: Optional[argparse.Namespace] = None,
        save_options: Optional[bool] = True,
    ):
        """
        Parse options.

        Args:
            opt: User-provided options.
            save_options: Whether to save the options to a text file.
        
        Returns:
            opt: Options, with the first GPU device set for PyTorch if the 
                device is available.
        """
        # If user doesn't provide options, parse options using argparse.
        if opt is None and not self.initialized:
            self.initialize()
            opt = self.parser.parse_args()

        # parse gpu id list
        str_gpu_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_gpu_ids:
            if str_id.isdigit() and int(str_id) >= 0:
                opt.gpu_ids.append(int(str_id))
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_ids[0])
        else:
            print("| using cpu")
            opt.gpu_ids = []

        # print and save options
        if save_options:
            opt.run_dir = os.path.join(opt.run_dir, util.get_time_dirname())
            args = vars(opt)
            print("| options")
            for k, v in args.items():
                print("%s: %s" % (str(k), str(v)))
            mkdirs(opt.run_dir)

            if self.is_train:
                filename = "train_opt.txt"
            else:
                filename = "test_opt.txt"
            file_path = os.path.join(opt.run_dir, filename)
            with open(file_path, "wt") as fout:
                fout.write("| options\n")
                for k, v in sorted(args.items()):
                    fout.write("%s: %s\n" % (str(k), str(v)))
        return opt


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            "--num_iters", default=250000, type=int, help="total number of iterations",
        )
        self.parser.add_argument(
            "--display_every",
            default=20,
            type=int,
            help="display training information every N iterations",
        )
        self.parser.add_argument(
            "--checkpoint_every",
            default=2000,
            type=int,
            help="save every N iterations",
        )
        self.parser.add_argument(
            "--shuffle_data", default=1, type=int, help="shuffle dataloader"
        )
        self.is_train = True


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            "--output_path",
            default="result.json",
            type=str,
            help="save path for derendered scene annotation",
        )
        self.parser.add_argument(
            "--clevr_val_ann_path",
            default="../../data/attr_net/objects/clevr_val_objs.json",
            type=str,
            help="clevr val objects annotation file",
        )
        self.parser.add_argument(
            "--clevr_val_img_dir",
            default="../../data/raw/CLEVR_v1.0/images/val",
            type=str,
            help="clevr val image directory",
        )
        self.parser.add_argument(
            "--shuffle_data", default=0, type=int, help="shuffle dataloader"
        )
        self.parser.add_argument(
            "--use_cat_label",
            default=0,
            type=int,
            help="use object detector class label",
        )
        self.is_train = False


def get_options(mode):
    if mode == "train":
        opt = TrainOptions().parse()
    elif mode == "test":
        opt = TestOptions().parse()
    else:
        raise ValueError("Invalid mode for option parsing: %s" % mode)
    return opt
