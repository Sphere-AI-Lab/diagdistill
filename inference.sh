export CUDA_VISIBLE_DEVICES=6
torchrun \
  --nproc_per_node=1 \
  --master_port=29668 \
  inference.py \
  --config_path configs/diadistill_inference.yaml