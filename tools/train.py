# Copyright (c) AlphaBetter. All rights reserved.
from conditional_gan.engine import Trainer, init_train_env
import argparse
from pathlib import Path

from omegaconf import OmegaConf

yaml_file_path = Path("../configs/baseline_mnist.yaml")


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        metavar="FILE",
        help="path to config file",
    )
    return parser.parse_args()


def main() -> None:
    opts = get_opts()

    config_dict = OmegaConf.load(opts.config_path)
    config_dict, device = init_train_env(config_dict)

    trainer = Trainer(config_dict, device)
    trainer.train()


if __name__ == "__main__":
    main()
