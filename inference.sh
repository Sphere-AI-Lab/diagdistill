#!/bin/bash

torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  inference.py \
  --config_path configs/diadistill_inference.yaml
