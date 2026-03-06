"""Video Editing Inference Script
=================================

This script performs text-driven video-to-video editing using the diadistill
causal video diffusion pipeline. Unlike the video continuation script, it
regenerates the entire input clip with controllable edit strength so that the
output video length matches the input length.
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.io import read_video, write_video

import peft
from pipeline import CausalInferencePipeline
from utils.lora_utils import configure_lora_for_model
from utils.misc import set_seed


def load_and_preprocess_video(
    video_path: str,
    target_frames: Optional[int] = None,
    target_size: tuple[int, int] = (480, 832),
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Load a video file and preprocess it for the model."""
    video_frames, _, _ = read_video(video_path, pts_unit="sec")
    video_frames = rearrange(video_frames, "t h w c -> t c h w")

    if target_frames is not None and video_frames.shape[0] > target_frames:
        video_frames = video_frames[:target_frames]

    resize_transform = transforms.Resize(target_size, antialias=True)
    video_frames = resize_transform(video_frames)

    video_frames = video_frames.float() / 255.0
    video_frames = (video_frames - 0.5) * 2.0
    video_frames = video_frames.unsqueeze(0)
    return video_frames.to(device)


def main() -> None:
    parser = argparse.ArgumentParser("Video Editing Inference")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
    parser.add_argument("--input_video", type=str, required=True, help="Path to video to edit")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the edit")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save edited video")
    parser.add_argument("--num_frames", type=int, default=None, help="Optional number of frames to edit")
    parser.add_argument(
        "--edit_strength",
        type=float,
        default=0.5,
        help="Edit strength in [0, 1]. 0 keeps input, 1 rewrites fully.",
    )
    parser.add_argument(
        "--context_noise",
        type=int,
        default=0,
        help="Noise level to write edited latents back into KV cache",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if not hasattr(config, "context_noise"):
        config.context_noise = args.context_noise

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        set_seed(config.seed + local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda")
        set_seed(config.seed)

    torch.set_grad_enabled(False)
    print(f"Running on device: {device}")

    print("Initializing pipeline...")
    pipeline = CausalInferencePipeline(config, device=device)

    if config.generator_ckpt:
        print(f"Loading checkpoint from {config.generator_ckpt}")
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        if config.use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")

            cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
            missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
            if missing:
                print(f"[Warning] {len(missing)} parameters missing")
            if unexpected:
                print(f"[Warning] {len(unexpected)} unexpected params")
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict)

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

    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    print("Models loaded successfully")

    print(f"\nLoading input video from {args.input_video}")
    input_video = load_and_preprocess_video(
        args.input_video,
        target_frames=args.num_frames,
        device=device,
    )

    batch_size, num_frames, _, height, width = input_video.shape
    print(f"Input video shape: {input_video.shape}")

    print("Encoding input video to latent space...")
    with torch.no_grad():
        video_for_vae = rearrange(input_video, "b t c h w -> b c t h w").to(dtype=torch.bfloat16)
        input_latents = pipeline.vae.encode_to_latent(video_for_vae).to(dtype=torch.bfloat16)

    print(f"Encoded latents shape: {input_latents.shape}")

    print(f"\nText prompt: '{args.prompt}'")
    conditional_dict = pipeline.text_encoder(text_prompts=[args.prompt])

    original_total_frames = input_latents.shape[1]
    num_frame_per_block = pipeline.num_frame_per_block

    if original_total_frames % num_frame_per_block != 0:
        pad_frames = num_frame_per_block - (original_total_frames % num_frame_per_block)
        last_frame = input_latents[:, -1:].repeat(1, pad_frames, 1, 1, 1)
        input_latents = torch.cat([input_latents, last_frame], dim=1)
        print(
            f"Padded input latents to {input_latents.shape[1]} frames "
            f"(multiple of {num_frame_per_block})"
        )
    else:
        pad_frames = 0

    total_frames = input_latents.shape[1]

    pipeline._initialize_kv_cache(
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=device,
        kv_cache_size_override=total_frames * pipeline.frame_seq_length,
    )
    pipeline._initialize_crossattn_cache(
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=device,
    )
    pipeline.generator.model.local_attn_size = pipeline.local_attn_size
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)

    denoising_steps = pipeline.denoising_step_list
    if not denoising_steps:
        raise RuntimeError("Pipeline denoising_step_list is empty")

    edit_strength = float(np.clip(args.edit_strength, 0.0, 1.0))
    start_index = int((1.0 - edit_strength) * (len(denoising_steps) - 1))
    start_index = max(0, min(start_index, len(denoising_steps) - 1))

    print(
        f"\nEditing video with strength {edit_strength:.2f} "
        f"(timestep index {start_index}, value {denoising_steps[start_index]})"
    )

    edited_latents = torch.zeros_like(input_latents)
    num_blocks = total_frames // num_frame_per_block

    current_start_frame = 0
    for block_idx in range(num_blocks):
        block_start = block_idx * num_frame_per_block
        block_end = block_start + num_frame_per_block

        base_block = input_latents[:, block_start:block_end]

        initial_timestep_value = denoising_steps[start_index]
        timestep_tensor = initial_timestep_value * torch.ones(
            [batch_size * num_frame_per_block],
            device=device,
            dtype=torch.long,
        )
        noisy_block = pipeline.scheduler.add_noise(
            base_block.flatten(0, 1),
            torch.randn_like(base_block.flatten(0, 1)),
            timestep_tensor,
        ).unflatten(0, base_block.shape[:2])

        noisy_input = noisy_block
        for step_idx in range(start_index, len(denoising_steps)):
            current_timestep = denoising_steps[step_idx]
            timestep = torch.ones(
                [batch_size, num_frame_per_block],
                device=device,
                dtype=torch.int64,
            ) * current_timestep

            is_last_step = step_idx == len(denoising_steps) - 1
            generator_outputs = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )

            _, denoised_pred = generator_outputs
            if not is_last_step:
                next_timestep = denoising_steps[step_idx + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size * num_frame_per_block],
                        device=device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        edited_latents[:, block_start:block_end] = denoised_pred

        context_timestep = torch.ones_like(timestep) * config.context_noise
        pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )

        current_start_frame += num_frame_per_block
        print(f"  Edited block {block_idx + 1}/{num_blocks}")

    print("\nDecoding edited latents to video...")
    with torch.no_grad():
        trimmed_latents = (
            edited_latents[:, :original_total_frames]
            if pad_frames > 0
            else edited_latents
        )
        decoded_video = pipeline.vae.decode_to_pixel(trimmed_latents, use_cache=False)
        decoded_video = (decoded_video * 0.5 + 0.5).clamp(0, 1)

    output_video = rearrange(decoded_video[0], "t c h w -> t h w c")
    output_video = (output_video.cpu() * 255.0).to(torch.uint8)

    print(f"\nSaving edited video to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    write_video(args.output_path, output_video, fps=16)

    print("\nVideo editing complete!")
    print(f"  Input frames: {num_frames}")
    print(f"  Output frames: {output_video.shape[0]}")
    print(f"  Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
