# Copyright (c) AlphaBetter. All rights reserved.
import argparse
import os
from pathlib import Path

import torch

from ECHO_ML.CGAN_Github_Lonatang.conditional_gan.engine.inferencer import Inferencer



def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="path to model weights",
    )
    parser.add_argument(
        "--device",
        default="0",
        type=str,
        help="device to use. Default is 0",
    )
    parser.add_argument(
        "-i", "--conditional-index",
        default=0,
        type=int,
        help="conditional index. Default is 0",
    )
    parser.add_argument(
        "--latent-dim",
        default=100,
        type=int,
        help="latent dimension. Default is 100",
    )
    parser.add_argument(
        "--num-samples",
        default=128,
        type=int,
        help="number of samples to generate. Default is 128",
    )
    parser.add_argument(
        "--save-path",
        default=Path("results/inference"),
        type=Path,
        help="path to save generated samples. Default is ``results/inference``",
    )
    return parser.parse_args()


def main() -> None:
    torch.set_float32_matmul_precision("high")

    opts = get_opts()

    inferencer = Inferencer(opts.weights, opts.device)

    for i in range(opts.num_samples):
        save_path = opts.save_path.joinpath(f"sample_{i:08d}.png")
        inferencer(save_path, opts.conditional_index, opts.latent_dim)


if __name__ == "__main__":
    main()
