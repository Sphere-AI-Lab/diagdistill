export CUDA_VISIBLE_DEVICES=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun \
  --nproc_per_node=1 \
  --master_port=30010 \
  interactive_inference.py \
  --config_path configs/diadistill_interactive_inference.yaml
