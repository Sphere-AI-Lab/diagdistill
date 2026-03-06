#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

CONFIG_PATH="configs/diadistill_inference.yaml"
INPUT_VIDEO="videos/interactive/rank0-0-0_lora.mp4"
OUTPUT_PATH="videos/interactive/rank0-0-0_lora_continuation.mp4"
PROMPT="A realistic continuation of a Texas Hold'em poker celebration at a casino, focusing on the same late-30s male player as he keeps celebrating among cheering patrons and glowing slot machines, with wide-to-medium close-up camera work."
NUM_CONTEXT_FRAMES=96
NUM_CONTINUATION_FRAMES=128
CONTEXT_NOISE=0

python3 video_continuation_inference.py \
  --config_path "${CONFIG_PATH}" \
  --input_video "${INPUT_VIDEO}" \
  --prompt "${PROMPT}" \
  --output_path "${OUTPUT_PATH}" \
  --num_context_frames "${NUM_CONTEXT_FRAMES}" \
  --num_continuation_frames "${NUM_CONTINUATION_FRAMES}" \
  --context_noise "${CONTEXT_NOISE}"
