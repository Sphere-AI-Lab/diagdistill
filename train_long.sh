#!/bin/bash
set -euo pipefail

# Always run from repo root so relative paths in configs/code are valid.
cd "$(dirname "$0")"

CONFIG=configs/diadistill_train_long.yaml
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="outputs"
LOGDIR="${OUTPUT_ROOT}/${RUN_TS}"
WANDB_SAVE_DIR="$LOGDIR"
MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 20000))}"
export WANDB_MODE=disabled

mkdir -p "$LOGDIR"
cp "$CONFIG" "$LOGDIR/train_config_input.yaml"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd=$(pwd)"
  echo "config=$CONFIG"
  echo "master_port=$MASTER_PORT"
  echo "command=torchrun --nproc_per_node=8 --master_port $MASTER_PORT train.py --config_path $CONFIG --logdir $LOGDIR --wandb-save-dir $WANDB_SAVE_DIR --disable-wandb"
} > "$LOGDIR/launch_info.txt"

echo "CONFIG=$CONFIG"
echo "RUN_DIR=$LOGDIR"
echo "MASTER_PORT=$MASTER_PORT"

torchrun \
  --nproc_per_node=8 \
  --master_port "$MASTER_PORT" \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$WANDB_SAVE_DIR" \
  --disable-wandb 2>&1 | tee "$LOGDIR/train.log"
