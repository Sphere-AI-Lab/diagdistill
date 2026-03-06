# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Video Continuation Inference: Given a video prefix, generate the continuation
# This leverages the autoregressive nature of the causal model to extend videos
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import argparse
import os
from typing import List, Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.io import read_video, write_video
from torchvision import transforms
from einops import rearrange
import numpy as np

from utils.misc import set_seed
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from pipeline import CausalInferencePipeline
from utils.lora_utils import configure_lora_for_model
import peft


def load_and_preprocess_video(
    video_path: str,
    target_frames: Optional[int] = None,
    target_size: tuple = (480, 832),
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Load a video file and preprocess it for the model.

    Args:
        video_path: Path to the video file
        target_frames: Number of frames to use (if None, uses all frames)
        target_size: Target (height, width) for resizing
        device: Device to load the video onto

    Returns:
        video_tensor: Tensor of shape (1, num_frames, 3, height, width) in range [0, 1]
    """
    # Read video
    video_frames, _, info = read_video(video_path, pts_unit='sec')

    # video_frames shape: (T, H, W, C) in range [0, 255]
    # Convert to (T, C, H, W)
    video_frames = rearrange(video_frames, 't h w c -> t c h w')

    # Select target number of frames
    if target_frames is not None and video_frames.shape[0] > target_frames:
        video_frames = video_frames[:target_frames]

    # Resize to target size
    resize_transform = transforms.Resize(target_size, antialias=True)
    video_frames = resize_transform(video_frames)

    # Normalize to [0, 1]
    video_frames = video_frames.float() / 255.0

    # Normalize to [-1, 1] for VAE encoding
    video_frames = (video_frames - 0.5) * 2.0

    # Add batch dimension: (1, T, C, H, W)
    video_frames = video_frames.unsqueeze(0)

    return video_frames.to(device)


def prefill_kv_cache_with_video(
    pipeline: CausalInferencePipeline,
    video_latents: torch.Tensor,
    conditional_dict: dict,
    context_noise_level: int = 0
) -> None:
    """
    Prefill the KV cache by running the model on the input video frames.
    This is the key to video continuation: we use the given video as context.

    Args:
        pipeline: The inference pipeline
        video_latents: Encoded video latents of shape (batch, num_frames, 16, H//8, W//8)
        conditional_dict: Text conditioning dictionary
        context_noise_level: Noise level to use for context (0 = clean)
    """
    batch_size, num_context_frames, num_channels, height, width = video_latents.shape
    device = video_latents.device

    print(f"Prefilling KV cache with {num_context_frames} context frames...")

    # Process each frame block to build up the KV cache
    current_start_frame = 0
    num_frame_per_block = pipeline.num_frame_per_block

    # Ensure num_context_frames is divisible by num_frame_per_block
    if num_context_frames % num_frame_per_block != 0:
        # Pad to nearest multiple
        pad_frames = num_frame_per_block - (num_context_frames % num_frame_per_block)
        # Replicate last frame for padding
        last_frame = video_latents[:, -1:].repeat(1, pad_frames, 1, 1, 1)
        video_latents = torch.cat([video_latents, last_frame], dim=1)
        num_context_frames = video_latents.shape[1]
        print(f"Padded context frames to {num_context_frames} (multiple of {num_frame_per_block})")

    num_blocks = num_context_frames // num_frame_per_block

    # Create context timestep (clean or slightly noisy)
    context_timestep_value = torch.tensor([context_noise_level], device=device)

    for block_idx in range(num_blocks):
        context_frames = video_latents[
            :, current_start_frame:current_start_frame + num_frame_per_block
        ]

        # Create timestep tensor for this block
        timestep = torch.ones(
            [batch_size, num_frame_per_block],
            device=device,
            dtype=torch.int64
        ) * context_noise_level

        # Run model to populate KV cache
        # We don't need the output, just want to fill the cache
        _ = pipeline.generator(
            noisy_image_or_video=context_frames,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length
        )

        current_start_frame += num_frame_per_block
        print(f"  Processed context block {block_idx + 1}/{num_blocks}")

    print(f"KV cache prefilled with {num_context_frames} frames")


# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("Video Continuation Inference")
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument("--input_video", type=str, required=True, help="Path to input video file")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt for continuation")
parser.add_argument("--output_path", type=str, required=True, help="Path to save output video")
parser.add_argument("--num_context_frames", type=int, default=None,
                    help="Number of frames from input video to use as context (default: use all)")
parser.add_argument("--num_continuation_frames", type=int, default=81,
                    help="Number of frames to generate as continuation")
parser.add_argument("--context_noise", type=int, default=0,
                    help="Noise level for context frames (0 = clean)")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Override some config values from command line
if not hasattr(config, 'context_noise'):
    config.context_noise = args.context_noise

# ----------------------------- Setup device -----------------------------
if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config.seed + local_rank)
else:
    local_rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)

print(f"Running on device: {device}")
torch.set_grad_enabled(False)

# ----------------------------- Load models -----------------------------
print("Initializing pipeline...")
pipeline = CausalInferencePipeline(config, device=device)

# Load checkpoint
if config.generator_ckpt:
    print(f"Loading checkpoint from {config.generator_ckpt}")
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]

    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if missing:
            print(f"[Warning] {len(missing)} parameters missing")
        if unexpected:
            print(f"[Warning] {len(unexpected)} unexpected params")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# LoRA support (optional)
pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    print(f"LoRA enabled with config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        print("LoRA weights loaded")

    pipeline.is_lora_enabled = True

# Move to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

print("Models loaded successfully")

# ----------------------------- Load and encode input video -----------------------------
print(f"\nLoading input video from {args.input_video}")
input_video = load_and_preprocess_video(
    args.input_video,
    target_frames=args.num_context_frames,
    device=device
)

batch_size, num_context_video_frames, _, height, width = input_video.shape
print(f"Input video shape: {input_video.shape} (frames: {num_context_video_frames})")

# Encode input video to latent space
print("Encoding input video to latent space...")
with torch.no_grad():
    # VAE wrapper expects (B, C, T, H, W) in bfloat16 to match weights
    video_for_vae = rearrange(input_video, 'b t c h w -> b c t h w').to(dtype=torch.bfloat16)
    input_latents = pipeline.vae.encode_to_latent(video_for_vae).to(dtype=torch.bfloat16)

num_context_latent_frames = input_latents.shape[1]
print(f"Encoded latents shape: {input_latents.shape}")

# ----------------------------- Prepare for continuation generation -----------------------------
# Get text conditioning
print(f"\nText prompt: '{args.prompt}'")
conditional_dict = pipeline.text_encoder(text_prompts=[args.prompt])

# Calculate total frames needed
total_frames = num_context_latent_frames + args.num_continuation_frames

# Ensure total_frames is divisible by num_frame_per_block
if total_frames % pipeline.num_frame_per_block != 0:
    pad_frames = pipeline.num_frame_per_block - (total_frames % pipeline.num_frame_per_block)
    total_frames += pad_frames
    print(f"Adjusted total frames to {total_frames} (multiple of {pipeline.num_frame_per_block})")

# Initialize KV cache
print("\nInitializing KV cache...")
local_attn_cfg = getattr(config.model_kwargs, "local_attn_size", -1)
if local_attn_cfg != -1:
    kv_cache_size = local_attn_cfg * pipeline.frame_seq_length
    print(f"Using local attention with cache size: {kv_cache_size}")
else:
    kv_cache_size = total_frames * pipeline.frame_seq_length
    print(f"Using global attention with cache size: {kv_cache_size}")

pipeline._initialize_kv_cache(
    batch_size=batch_size,
    dtype=torch.bfloat16,
    device=device,
    kv_cache_size_override=kv_cache_size
)
pipeline._initialize_crossattn_cache(
    batch_size=batch_size,
    dtype=torch.bfloat16,
    device=device
)

pipeline.generator.model.local_attn_size = pipeline.local_attn_size
pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)

# Prefill KV cache with input video
prefill_kv_cache_with_video(
    pipeline=pipeline,
    video_latents=input_latents,
    conditional_dict=conditional_dict,
    context_noise_level=config.context_noise
)

# ----------------------------- Generate continuation -----------------------------
print(f"\nGenerating {args.num_continuation_frames} continuation frames...")

# Create noise for continuation frames
continuation_noise = torch.randn(
    [batch_size, args.num_continuation_frames, 16,
     input_latents.shape[3], input_latents.shape[4]],
    device=device,
    dtype=torch.bfloat16
)

# Pad if necessary
if continuation_noise.shape[1] < (total_frames - num_context_latent_frames):
    pad_frames = total_frames - num_context_latent_frames - continuation_noise.shape[1]
    continuation_noise = torch.cat([
        continuation_noise,
        torch.randn([batch_size, pad_frames, 16,
                    input_latents.shape[3], input_latents.shape[4]],
                   device=device, dtype=torch.bfloat16)
    ], dim=1)

# Allocate output tensor for continuation
output_latents = torch.zeros_like(continuation_noise)

# Start frame index (where continuation begins)
current_start_frame = num_context_latent_frames
num_continuation_blocks = continuation_noise.shape[1] // pipeline.num_frame_per_block

print(f"Generating {num_continuation_blocks} continuation blocks...")

# Generate continuation autoregressively
for block_idx in range(num_continuation_blocks):
    block_start = block_idx * pipeline.num_frame_per_block
    block_end = block_start + pipeline.num_frame_per_block

    noisy_input = continuation_noise[:, block_start:block_end]

    # Denoising loop for this block
    for step_idx, current_timestep in enumerate(pipeline.denoising_step_list):
        timestep = torch.ones(
            [batch_size, pipeline.num_frame_per_block],
            device=device,
            dtype=torch.int64
        ) * current_timestep

        if step_idx < len(pipeline.denoising_step_list) - 1:
            # Intermediate denoising step
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length
            )

            # Add noise for next step
            next_timestep = pipeline.denoising_step_list[step_idx + 1]
            noisy_input = pipeline.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                next_timestep * torch.ones(
                    [batch_size * pipeline.num_frame_per_block],
                    device=device,
                    dtype=torch.long
                )
            ).unflatten(0, denoised_pred.shape[:2])
        else:
            # Final denoising step
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length
            )

    # Store output
    output_latents[:, block_start:block_end] = denoised_pred

    # Update KV cache with clean output for next block
    context_timestep = torch.ones_like(timestep) * config.context_noise
    pipeline.generator(
        noisy_image_or_video=denoised_pred,
        conditional_dict=conditional_dict,
        timestep=context_timestep,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start_frame * pipeline.frame_seq_length
    )

    current_start_frame += pipeline.num_frame_per_block
    print(f"  Generated continuation block {block_idx + 1}/{num_continuation_blocks}")

# Trim to actual continuation length
output_latents = output_latents[:, :args.num_continuation_frames]

# ----------------------------- Decode to video -----------------------------
print("\nDecoding latents to video...")
with torch.no_grad():
    # Combine context and continuation latents
    full_latents = torch.cat([input_latents, output_latents], dim=1)

    # Decode
    full_video = pipeline.vae.decode_to_pixel(full_latents, use_cache=False)
    full_video = (full_video * 0.5 + 0.5).clamp(0, 1)
    total_output_video_frames = full_video.shape[1]

generated_video_frames = max(total_output_video_frames - num_context_video_frames, 0)

# Convert to uint8 for saving
output_video = rearrange(full_video[0], 't c h w -> t h w c')
output_video = (output_video.cpu() * 255.0).to(torch.uint8)

# ----------------------------- Save output -----------------------------
print(f"\nSaving output video to {args.output_path}")
os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
write_video(args.output_path, output_video, fps=16)

print(f"\nVideo continuation complete!")
print(f"  Input video frames: {num_context_video_frames}")
print(f"  Input latent frames: {num_context_latent_frames}")
print(f"  Generated latent frames: {args.num_continuation_frames}")
print(f"  Total output latent frames: {num_context_latent_frames + args.num_continuation_frames}")
print(f"  Output video frames: {total_output_video_frames}")
print(f"  Generated video frames: {generated_video_frames}")
print(f"  Output saved to: {args.output_path}")
