torchrun --nproc_per_node=8 --nnodes=1 --master_port 29503 \
    train.py \
    --config_path  \
    --logdir  \
    --disable-wandb