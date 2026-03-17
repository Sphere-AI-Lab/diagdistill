# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import shlex
import sys
from omegaconf import OmegaConf
import wandb

from trainer import ScoreDistillationTrainer


def _load_shared_schedule(config):
    schedule_path = config.get("schedule_config_path", "")
    if not schedule_path:
        return config
    shared_schedule = OmegaConf.load(schedule_path)
    # Schedule is the single source of truth for train/infer denoising settings.
    return OmegaConf.merge(config, shared_schedule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--no-one-logger", action="store_true", help="Disable One Logger (enabled by default)")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config = _load_shared_schedule(config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    # config_name = os.path.basename(args.config_path).split(".")[0]
    config_name = os.path.dirname(args.config_path).split("/")[-1]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    config.use_one_logger = not args.no_one_logger
    config.config_path = os.path.abspath(args.config_path)
    config.launch_command = " ".join(shlex.quote(arg) for arg in sys.argv)
    if config.logdir:
        os.makedirs(config.logdir, exist_ok=True)
        resolved_path = os.path.join(config.logdir, "training_config_resolved.yaml")
        OmegaConf.save(config=config, f=resolved_path, resolve=True)

    if config.trainer == "score_distillation":
        trainer = ScoreDistillationTrainer(config)
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
