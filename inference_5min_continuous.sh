#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
NUM_SEGMENTS=${NUM_SEGMENTS:-10}
SEGMENT_FRAMES=${SEGMENT_FRAMES:-120}
PROMPT_INDEX=${PROMPT_INDEX:-0}
FPS=${FPS:-16}
OUTPUT_NAME=${OUTPUT_NAME:-}
CONTEXT_FRAMES=${CONTEXT_FRAMES:-}
CONFIG_PATH=${CONFIG_PATH:-configs/diadistill_inference.yaml}

CMD=(
  python inference_5min_continuous.py
  --config_path "$CONFIG_PATH"
  --num_segments "$NUM_SEGMENTS"
  --segment_frames "$SEGMENT_FRAMES"
  --prompt_index "$PROMPT_INDEX"
  --fps "$FPS"
)

if [[ -n "$OUTPUT_NAME" ]]; then
  CMD+=(--output_name "$OUTPUT_NAME")
fi

if [[ -n "$CONTEXT_FRAMES" ]]; then
  CMD+=(--context_frames "$CONTEXT_FRAMES")
fi

"${CMD[@]}"
