# #!/bin/bash

# Project path and config
CONFIG=configs/diadistill_train_init.yaml
LOGDIR=logs
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_MODE=disabled
torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR