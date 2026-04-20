"""
Name: Agam Sidhu
USC ID: 3027948957
USC Email: agamsidh@usc.edu
Submission Date: April 19th, 2026
Entry Point: Problems 1(b), 1(c), 1(d), and 2(b)
"""

import argparse

from lenet import run_lenet_experiment, run_noise_experiment
from vit import run_vit_experiment


# Argument Parsing

def build_parser():
    parser = argparse.ArgumentParser(description="EE569 HW5 homework runner")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["mnist", "fashion", "cifar", "noise", "vit"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. LeNet uses dataset-specific defaults; ViT defaults to 20.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per hyperparameter setting.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for training and testing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--setting",
        default="all",
        help="Use all settings, a 1-based index such as 1, or a name such as setting_2.",
    )
    parser.add_argument(
        "--noise-dataset",
        default="fashion",
        choices=["mnist", "fashion", "cifar"],
        help="Dataset used for the noisy-label experiment. HW5 Problem 1(d) requires fashion; mnist/cifar are optional side experiments.",
    )
    return parser


# Main Execution

def main():
    args = build_parser().parse_args()

    if args.experiment in {"mnist", "fashion", "cifar"}:
        run_lenet_experiment(args.experiment, args)
    elif args.experiment == "noise":
        run_noise_experiment(args)
    else:
        run_vit_experiment(args)


if __name__ == "__main__":
    main()
