"""Batch continuous video generation without visible cuts.

This script iterates over all prompts (like inference.sh) but for each prompt it
invokes the continuation-based generator so that segments remain temporally
coherent.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch continuous inference")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--segment_frames", type=int, default=None)
    parser.add_argument("--context_frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--max_prompts", type=int, default=None)
    return parser.parse_args()


def init_device_and_seed(config) -> Tuple[torch.device, int, int]:
    if "LOCAL_RANK" in os.environ:
        os.environ.setdefault("NCCL_CROSS_NIC", "1")
        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_TIMEOUT", "1800")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", str(local_rank)))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.constants.default_pg_timeout,
            )
        set_seed(config.seed + local_rank)
    else:
        local_rank = 0
        rank = 0
        torch.cuda.set_device(0)
        set_seed(config.seed)
    return torch.device(f"cuda:{local_rank}"), local_rank, rank


def prepare_pipeline(config, device: torch.device, local_rank: int) -> CausalInferencePipeline:
    pipeline = CausalInferencePipeline(config, device=device)
    if config.generator_ckpt:
        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
        if "generator" in state_dict or "generator_ema" in state_dict:
            raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
        elif "model" in state_dict:
            raw_gen_state_dict = state_dict["model"]
        else:
            raise ValueError(f"Unexpected checkpoint keys in {config.generator_ckpt}")
        if config.use_ema:
            def _clean_key(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")

            cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
            pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        else:
            pipeline.generator.load_state_dict(raw_gen_state_dict)

    from utils.lora_utils import configure_lora_for_model
    import peft

    if getattr(config, "adapter", None) and configure_lora_for_model is not None:
        if local_rank == 0:
            print(f"LoRA enabled with config: {config.adapter}")
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=(local_rank == 0),
        )
        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path:
            lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore[arg-type]
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore[arg-type]

    pipeline = pipeline.to(dtype=torch.bfloat16)
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    return pipeline


def sanitize_prompt(prompt: str, idx: int) -> str:
    fragment = prompt[:50]
    sanitized = ''.join('_' if ch.isspace() else ch for ch in fragment)
    return sanitized or f"prompt_{idx}"


def decode_latents_to_video(pipeline, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    video = pipeline.vae.decode_to_pixel(latents.to(device), use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = rearrange(video[0], 't c h w -> t h w c').cpu()
    return (video * 255.0).to(torch.uint8)


def prefill_kv_cache(pipeline, context_latents: torch.Tensor, conditional_dict: dict, context_noise: int) -> None:
    batch_size, num_frames, _, _, _ = context_latents.shape
    num_frame_per_block = pipeline.num_frame_per_block
    device = context_latents.device

    if num_frames % num_frame_per_block != 0:
        pad = num_frame_per_block - (num_frames % num_frame_per_block)
        pad_frames = context_latents[:, -1:].repeat(1, pad, 1, 1, 1)
        context_latents = torch.cat([context_latents, pad_frames], dim=1)
        num_frames = context_latents.shape[1]

    num_blocks = num_frames // num_frame_per_block
    context_timestep = torch.ones([batch_size, num_frame_per_block], device=device, dtype=torch.int64) * context_noise
    current_start = 0

    for block_idx in range(num_blocks):
        frames = context_latents[:, block_idx * num_frame_per_block:(block_idx + 1) * num_frame_per_block]
        pipeline.generator(
            noisy_image_or_video=frames,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start * pipeline.frame_seq_length,
        )
        current_start += num_frame_per_block


def generate_continuation_latents(
    pipeline,
    conditional_dict: dict,
    context_latents: torch.Tensor,
    continuation_frames: int,
    context_noise: int,
) -> torch.Tensor:
    batch_size, num_context_frames, _, height, width = context_latents.shape
    device = context_latents.device

    total_frames = num_context_frames + continuation_frames
    if total_frames % pipeline.num_frame_per_block != 0:
        pad = pipeline.num_frame_per_block - (total_frames % pipeline.num_frame_per_block)
        total_frames += pad

    local_attn_cfg = getattr(pipeline.args.model_kwargs, "local_attn_size", -1)
    if local_attn_cfg != -1:
        kv_cache_size = local_attn_cfg * pipeline.frame_seq_length
    else:
        kv_cache_size = total_frames * pipeline.frame_seq_length

    pipeline._initialize_kv_cache(
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=device,
        kv_cache_size_override=kv_cache_size,
    )
    pipeline._initialize_crossattn_cache(
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=device,
    )
    pipeline.generator.model.local_attn_size = pipeline.local_attn_size
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)

    prefill_kv_cache(pipeline, context_latents, conditional_dict, context_noise)

    noise = torch.randn(
        [batch_size, continuation_frames, 16, height, width],
        device=device,
        dtype=torch.bfloat16,
    )
    if continuation_frames % pipeline.num_frame_per_block != 0:
        pad = pipeline.num_frame_per_block - (continuation_frames % pipeline.num_frame_per_block)
        extra = torch.randn([batch_size, pad, 16, height, width], device=device, dtype=torch.bfloat16)
        noise = torch.cat([noise, extra], dim=1)

    output_latents = torch.zeros_like(noise)
    current_start = num_context_frames
    num_blocks = noise.shape[1] // pipeline.num_frame_per_block

    for block_idx in range(num_blocks):
        block_start = block_idx * pipeline.num_frame_per_block
        block_end = block_start + pipeline.num_frame_per_block
        noisy_input = noise[:, block_start:block_end]

        for step_idx, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones(
                [batch_size, pipeline.num_frame_per_block],
                device=device,
                dtype=torch.int64,
            ) * current_timestep

            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start * pipeline.frame_seq_length,
            )

            if step_idx < len(pipeline.denoising_step_list) - 1:
                next_timestep = pipeline.denoising_step_list[step_idx + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size * pipeline.num_frame_per_block],
                        device=device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        output_latents[:, block_start:block_end] = denoised_pred
        context_timestep = torch.ones_like(timestep) * context_noise
        pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start * pipeline.frame_seq_length,
        )
        current_start += pipeline.num_frame_per_block

    return output_latents[:, :continuation_frames]


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    segment_frames = args.segment_frames or config.num_output_frames
    config.num_output_frames = segment_frames
    context_frames = args.context_frames or segment_frames

    device, local_rank, rank = init_device_and_seed(config)
    config.distributed = dist.is_initialized()
    free_vram = get_cuda_free_memory_gb(device)
    low_memory = free_vram < 40
    if rank == 0:
        print(f"[cont-batch] Device {device}, free VRAM ~{free_vram:.1f} GB, low_memory={low_memory}")

    torch.set_grad_enabled(False)
    pipeline = prepare_pipeline(config, device, local_rank)

    extended_prompt_path = getattr(config, "extended_data_path", None)
    if extended_prompt_path is None and hasattr(config, "extended_prompt_path"):
        extended_prompt_path = config.extended_prompt_path
    if extended_prompt_path is None:
        extended_prompt_path = config.data_path
    dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)

    total_prompts = len(dataset)
    start = max(0, args.start_index)
    end = total_prompts - 1 if args.end_index < 0 else min(args.end_index, total_prompts - 1)
    if start > end:
        raise ValueError("start_index must be <= end_index")
    if args.max_prompts is not None:
        end = min(end, start + args.max_prompts - 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config.output_folder, timestamp)
    if rank == 0:
        os.makedirs(run_output_dir, exist_ok=True)

    filename_counts: Dict[str, int] = {}

    for idx in range(start, end + 1):
        sample = dataset[idx]
        prompt = sample["prompts"]
        extended_prompt = sample.get("extended_prompts")
        prompts: List[str] = [extended_prompt or prompt] * config.num_samples
        conditional_dict = pipeline.text_encoder(text_prompts=prompts)

        if rank == 0:
            print(f"[cont-batch] ({idx}) generating continuous video")

        noise = torch.randn(
            [config.num_samples, segment_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )
        video, latents = pipeline.inference(
            noise=noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
            profile=False,
        )
        current_latents = latents.cpu()
        video_segments = [
            (255.0 * rearrange(video, 'b t c h w -> b t h w c').cpu()).to(torch.uint8)[0]
        ]

        for seg in range(1, args.num_segments):
            if rank == 0:
                print(f"[cont-batch]   continuation segment {seg + 1}/{args.num_segments}")
            context = current_latents[:, -context_frames:].to(device)
            new_latents = generate_continuation_latents(
                pipeline,
                conditional_dict,
                context,
                continuation_frames=segment_frames,
                context_noise=config.context_noise,
            )
            current_latents = torch.cat([current_latents, new_latents.cpu()], dim=1)
            decoded = decode_latents_to_video(pipeline, new_latents, device)
            video_segments.append(decoded)
            torch.cuda.empty_cache()

        full_video = torch.cat(video_segments, dim=0)

        base_name = sanitize_prompt(prompt, idx)
        filename = base_name
        count = filename_counts.get(base_name, 0)
        filename_counts[base_name] = count + 1
        if count > 0:
            filename = f"{base_name}_{count}"
        output_path = os.path.join(run_output_dir, f"{filename}.mp4")
        write_video(output_path, full_video, fps=args.fps)

        if rank == 0:
            print(f"[cont-batch]   saved {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
