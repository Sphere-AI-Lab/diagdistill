# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import argparse
import torch
import os
import urllib.request
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from datetime import datetime
import time

from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset
from utils.misc import set_seed

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)

def initialize_vae_decoder(use_taehv: bool, taehv_checkpoint_path: str, device: torch.device):
    if not use_taehv:
        return None

    import sys
    if "/home/liujinxiu/new_self-forcing" not in sys.path:
        sys.path.insert(0, "/home/liujinxiu/new_self-forcing")
    from demo_utils.taehv import TAEHV

    if not os.path.exists(taehv_checkpoint_path):
        print(f"taew2_1.pth not found at {taehv_checkpoint_path}. Downloading...")
        os.makedirs(os.path.dirname(taehv_checkpoint_path), exist_ok=True)
        download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
        urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
        print(f"Downloaded TAEHV checkpoint to {taehv_checkpoint_path}")

    class TinyVAEWrapper(torch.nn.Module):
        def __init__(self, checkpoint_path: str):
            super().__init__()
            self.taehv = TAEHV(checkpoint_path=checkpoint_path).to(torch.float16)

        def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
            # TAEHV expects NTCHW and returns NTCHW in ~[0,1], convert to ~[-1,1] to match WanVAEWrapper.
            del use_cache
            latents_fp16 = latent.to(dtype=torch.float16)
            return self.taehv.decode_video(latents_fp16, parallel=False).mul_(2).sub_(1)

    tiny_vae = TinyVAEWrapper(taehv_checkpoint_path).eval().requires_grad_(False)
    tiny_vae.to(device)
    print("TAEHV tiny decoder enabled for inference")
    return tiny_vae


# Initialize pipeline
# Note: checkpoint loading is now handled inside the pipeline __init__ method
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # 在加载基础权重后，对 generator 的 transformer 模型应用 LoRA 包装
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # 加载 LoRA 权重（如果提供了 lora_ckpt）
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # 兼容包含 `generator_lora` 键或直接是 LoRA state dict 两种格式
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True


# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

use_taehv = bool(getattr(config, "use_taehv", False))
taehv_checkpoint_path = str(getattr(config, "taehv_checkpoint_path", "checkpoints/taew2_1.pth"))
if use_taehv:
    tiny_vae = initialize_vae_decoder(use_taehv=True, taehv_checkpoint_path=taehv_checkpoint_path, device=device)
    if tiny_vae is not None:
        pipeline.vae = tiny_vae

extended_prompt_path = config.data_path
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Build a run-specific output folder based on current time so each run is isolated.
if dist.is_initialized():
    timestamp_container = [datetime.now().strftime("%Y%m%d_%H%M%S") if local_rank == 0 else None]
    dist.broadcast_object_list(timestamp_container, src=0)
    run_timestamp = timestamp_container[0]
else:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

run_output_dir = os.path.join(config.output_folder, run_timestamp)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(run_output_dir, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()
    video_start_time = datetime.now()
    total_preprocessing_time = 0.0
    total_denoising_time = 0.0
    total_vae_processing_time = 0.0

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    preprocess_start = time.perf_counter()
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )
    preprocess_end = time.perf_counter()
    total_preprocessing_time += (preprocess_end - preprocess_start)

    print("sampled_noise.device", sampled_noise.device)
    # print("initial_latent.device", initial_latent.device)
    print("prompts", prompts)
    # Generate 81 frames
    # print('sampled_noise.shape', sampled_noise.shape, 'prompts', prompts)
    # print('pipeline.generator', pipeline.generator)
    # print('pipeline.text_encoder', pipeline.text_encoder)
    # print('pipeline.vae', pipeline.vae)

    denoising_start = time.perf_counter()
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
    )
    denoising_end = time.perf_counter()
    total_denoising_time += (denoising_end - denoising_start)

    vae_processing_start = time.perf_counter()
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache when available (WanVAEWrapper has model.clear_cache; TAEHV wrapper does not).
    if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
        pipeline.vae.model.clear_cache()
    vae_processing_end = time.perf_counter()
    total_vae_processing_time += (vae_processing_end - vae_processing_start)

    end_time = datetime.now()
    total_duration = end_time - video_start_time

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    saved_video_path = None
    final_export_start = time.perf_counter()
    if idx < num_prompts:
        base_prompt_fragment = prompt[:50]
        sanitized_name = ''.join('_' if ch.isspace() else ch for ch in base_prompt_fragment)
        if not sanitized_name:
            sanitized_name = f"prompt_{idx}"

        for seed_idx in range(config.num_samples):
            # Ensure filenames stay unique even if prompts repeat
            filename = f"{sanitized_name}.mp4"
            output_path = os.path.join(run_output_dir, filename)
            duplicate_idx = 1
            while os.path.exists(output_path):
                filename = f"{sanitized_name}_{duplicate_idx}.mp4"
                output_path = os.path.join(run_output_dir, filename)
                duplicate_idx += 1
            write_video(output_path, video[seed_idx], fps=16)
            if saved_video_path is None:
                saved_video_path = output_path
    final_export_end = time.perf_counter()
    final_export_time = final_export_end - final_export_start

    if rank == 0:
        block_timing = getattr(pipeline, "last_block_timing", [])
        total_seconds = max(total_duration.total_seconds(), 1e-8)
        denoising_seconds = max(total_denoising_time, 1e-8)
        print(f"\n{'='*60}")
        print("DETAILED TIMING REPORT")
        print(f"{'='*60}")
        print(f"Total execution time: {total_duration}")
        print(f"Total preprocessing time: {total_preprocessing_time*1000:.2f} ms ({total_preprocessing_time/total_seconds*100:.1f}%)")
        print(f"Total PURE DENOISING time: {total_denoising_time*1000:.2f} ms ({total_denoising_time/total_seconds*100:.1f}%)")
        print(f"Total VAE processing time: {total_vae_processing_time*1000:.2f} ms ({total_vae_processing_time/total_seconds*100:.1f}%)")
        if block_timing:
            print("Per-chunk denoising time:")
            for rec in block_timing:
                print(
                    f"  - chunk {rec['block_index'] + 1}: {rec['time_ms']:.2f} ms "
                    f"(steps={rec['step_count']})"
                )
            two_step_chunks = [rec for rec in block_timing if rec["step_count"] == 2]
            if two_step_chunks:
                print("Last 5 two-step chunk times:")
                for rec in two_step_chunks[-5:]:
                    print(f"  - chunk {rec['block_index'] + 1}: {rec['time_ms']:.2f} ms")
        print("")
        print(f"Average denoising time per rollout: {total_denoising_time*1000:.2f} ms")
        print(f"Denoising FPS: {1.0/denoising_seconds:.2f} rollouts/second")
        print(f"{'='*60}")
        print(f"Final video export time: {final_export_time*1000:.2f} ms")
        if saved_video_path is not None:
            print(f"Video saved to: {saved_video_path}")
        print(f"\nSUMMARY: Pure denoising took {total_denoising_time*1000:.2f} ms out of total {total_duration.total_seconds()*1000:.2f} ms")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
if dist.is_initialized():
    dist.destroy_process_group()
