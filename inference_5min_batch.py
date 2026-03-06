"""Batch runner that iterates over every prompt and generates ~5-minute videos.

It mirrors the original inference saving layout (output/<timestamp>/prompt.mp4),
but internally stitches multiple standard inference calls per prompt so that the
Wan model never exceeds its rotary embedding limits.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Tuple

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
    parser = argparse.ArgumentParser(description="Batch 5-minute inference")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--segment_frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="Inclusive end index; -1 means process to the end of the file.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional cap on number of prompts to process (starting from start_index).",
    )
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
            raise ValueError(f"Unexpected keys in {config.generator_ckpt}")
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


def ensure_unique_name(base: str, existing: Dict[str, int]) -> str:
    count = existing.get(base, 0)
    existing[base] = count + 1
    if count == 0:
        return base
    return f"{base}_{count}"


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    segment_frames = args.segment_frames or config.num_output_frames
    config.num_output_frames = segment_frames

    device, local_rank, rank = init_device_and_seed(config)
    config.distributed = dist.is_initialized()
    free_vram = get_cuda_free_memory_gb(device)
    low_memory = free_vram < 40
    if rank == 0:
        print(f"[5min-batch] Device {device}, free VRAM ~{free_vram:.1f} GB, low_memory={low_memory}")

    torch.set_grad_enabled(False)
    pipeline = prepare_pipeline(config, device, local_rank)

    extended_prompt_path = getattr(config, "extended_data_path", None)
    if extended_prompt_path is None and hasattr(config, "extended_prompt_path"):
        extended_prompt_path = config.extended_prompt_path
    if extended_prompt_path is None:
        extended_prompt_path = config.data_path
    dataset = TextDataset(
        prompt_path=config.data_path,
        extended_prompt_path=extended_prompt_path,
    )

    total_prompts = len(dataset)
    start = max(0, args.start_index)
    end = total_prompts - 1 if args.end_index < 0 else min(args.end_index, total_prompts - 1)
    if start > end:
        raise ValueError("start_index must be <= end_index")
    if args.max_prompts is not None:
        end = min(end, start + args.max_prompts - 1)

    if rank == 0:
        print(f"[5min-batch] Processing prompts {start} to {end} (inclusive)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config.output_folder, timestamp)
    if rank == 0:
        os.makedirs(run_output_dir, exist_ok=True)

    filename_counts: Dict[str, int] = {}

    for idx in range(start, end + 1):
        sample = dataset[idx]
        prompt = sample["prompts"]
        extended_prompt = sample.get("extended_prompts")
        prompts = [extended_prompt or prompt] * config.num_samples
        base_name = sanitize_prompt(prompt, idx)
        unique_name = ensure_unique_name(base_name, filename_counts)
        output_path = os.path.join(run_output_dir, f"{unique_name}_5min.mp4")

        if rank == 0:
            print(f"[5min-batch] ({idx}) generating -> {output_path}")

        chunks = []
        for seg in range(args.num_segments):
            if rank == 0:
                print(f"[5min-batch]   segment {seg + 1}/{args.num_segments}")
            noise = torch.randn(
                [config.num_samples, segment_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )
            video = pipeline.inference(
                noise=noise,
                text_prompts=prompts,
                return_latents=False,
                low_memory=low_memory,
                profile=False,
            )
            chunks.append(rearrange(video, "b t c h w -> b t h w c").cpu())
            pipeline.vae.model.clear_cache()

        full_video = (255.0 * torch.cat(chunks, dim=1)).clamp(0, 255).to(torch.uint8)
        for seed_idx in range(config.num_samples):
            write_video(output_path, full_video[seed_idx], fps=args.fps)
        torch.cuda.empty_cache()

        if rank == 0:
            print(f"[5min-batch]   saved {output_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
