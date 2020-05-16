"""Plots loss curves.

This script expects the provided `args.losses_path` to be a JSON file with the 
following format:
        {
            "train_losses": [
                <float>,
                ...
            ],
            "train_losses_ts": [
                <float>,
                ...
            ],
            "val_losses": [
                <float>,
                ...
            ],
            "val_losses_ts": [
                <float>,
                ...
            ],
        }

The script will generate a PNG file, saved to <args.out_path>, of a 2D line 
plot, where the X axis is the training iteration timestep, and the Y axis is 
the loss value. There will be two lines, a training line and a validation line.
"""
import os
import argparse
import matplotlib.pyplot as plt

import ns_vqa_dart.bullet.util as util


def main(args: argparse.Namespace):
    # Load the JSON file.
    load_path = os.path.join(args.run_dir, "stats.json")
    save_path = os.path.join(args.run_dir, "loss.png")
    loss_dict = util.load_json(path=load_path)

    train_ts = loss_dict["train_losses_ts"]
    train_losses = loss_dict["train_losses"]
    val_ts = loss_dict["val_losses_ts"]
    val_losses = loss_dict["val_losses"]

    plt.yscale("log")
    plt.plot(train_ts, train_losses, label="Training Loss")
    plt.plot(val_ts, val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        type=str,
        help="The path to the JSON file containing training and validation losses for plotting.",
    )
    args = parser.parse_args()
    main(args=args)
