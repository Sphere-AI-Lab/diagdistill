#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
NUM_SEGMENTS=${NUM_SEGMENTS:-10}
SEGMENT_FRAMES=${SEGMENT_FRAMES:-120}
FPS=${FPS:-16}
START_INDEX=${START_INDEX:-0}
END_INDEX=${END_INDEX:--1}
MAX_PROMPTS=${MAX_PROMPTS:-}
CONFIG_PATH=${CONFIG_PATH:-configs/diadistill_inference.yaml}

CMD=(
  python inference_5min_batch.py
  --config_path "$CONFIG_PATH"
  --num_segments "$NUM_SEGMENTS"
  --segment_frames "$SEGMENT_FRAMES"
  --fps "$FPS"
  --start_index "$START_INDEX"
  --end_index "$END_INDEX"
)

if [[ -n "$MAX_PROMPTS" ]]; then
  CMD+=(--max_prompts "$MAX_PROMPTS")
fi

"${CMD[@]}"
