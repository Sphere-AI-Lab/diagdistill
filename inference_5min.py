"""Utility script to generate longer (e.g., 5-minute) videos without touching the
original inference entrypoint.

The core idea is to reuse the existing `CausalInferencePipeline` but run it
multiple times per prompt, each time producing the usual `config.num_output_frames`
latent frames (~30 seconds at 16 FPS). The decoded chunks are concatenated on the
time dimension before writing a single MP4.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.memory import (
    DynamicSwapInstaller,
    get_cuda_free_memory_gb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-segment videos")
    parser.add_argument("--config_path", required=True, help="Base config file")
    parser.add_argument(
        "--num_segments",
        type=int,
        default=10,
        help="How many standard inference chunks to stitch (10≈5 minutes).",
    )
    parser.add_argument(
        "--segment_frames",
        type=int,
        default=None,
        help="Latent frames per chunk; defaults to config.num_output_frames.",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=0,
        help="Index inside the prompts file to render.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for the final MP4.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional explicit MP4 name (without path).",
    )
    return parser.parse_args()


def init_device_and_seed(config) -> tuple[torch.device, int, int]:
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


def select_prompt(dataset: TextDataset, idx: int) -> tuple[List[str], str]:
    assert 0 <= idx < len(dataset), f"prompt_index {idx} out of range"
    sample = dataset[idx]
    prompt = sample["prompts"]
    extended = sample.get("extended_prompts")
    if extended:
        prompts = [extended]
    else:
        prompts = [prompt]
    return prompts, prompt


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config_path)

    segment_frames = args.segment_frames or config.num_output_frames
    assert segment_frames > 0
    config.num_output_frames = segment_frames

    device, local_rank, rank = init_device_and_seed(config)
    config.distributed = dist.is_initialized()
    free_vram = get_cuda_free_memory_gb(device)
    low_memory = free_vram < 40
    if rank == 0:
        print(f"[5min] Using device {device}, free VRAM ~{free_vram:.1f} GB, low_memory={low_memory}")

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
    prompts, base_prompt = select_prompt(dataset, args.prompt_index)
    prompts = prompts * config.num_samples

    all_chunks = []
    for seg in range(args.num_segments):
        if rank == 0:
            print(f"[5min] Generating segment {seg + 1}/{args.num_segments} ...")
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
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu()
        all_chunks.append(current_video)
        pipeline.vae.model.clear_cache()

    full_video = (255.0 * torch.cat(all_chunks, dim=1)).clamp(0, 255).to(torch.uint8)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(config.output_folder, timestamp)
    if rank == 0:
        os.makedirs(run_output_dir, exist_ok=True)

    sanitized = ''.join('_' if ch.isspace() else ch for ch in base_prompt[:50]) or f"prompt_{args.prompt_index}"
    filename = args.output_name or f"{sanitized}_5min.mp4"
    output_path = os.path.join(run_output_dir, filename)

    for seed_idx in range(config.num_samples):
        write_video(output_path, full_video[seed_idx], fps=args.fps)
        print(f"[5min] Saved {output_path}")


if __name__ == "__main__":
    main()
