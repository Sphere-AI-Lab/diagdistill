"""Real-time diadistill demo that mirrors the new_self-forcing web experience."""

from __future__ import annotations

import argparse
import base64
import os
import queue
import time
from datetime import datetime
from io import BytesIO
from threading import Thread, Event
from typing import Optional

import numpy as np
import torch
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
from omegaconf import OmegaConf
from PIL import Image

from pipeline import CausalInferencePipeline
from utils.lora_utils import configure_lora_for_model
from utils.memory import (
    DynamicSwapInstaller,
    get_cuda_free_memory_gb,
    gpu,
    move_model_to_device_with_memory_preservation,
)
from utils.misc import set_seed

import peft

parser = argparse.ArgumentParser(description="diadistill real-time demo")
parser.add_argument('--port', type=int, default=5013)
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument('--config_path', type=str, default='configs/diadistill_inference.yaml')
args = parser.parse_args()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diadistill_realtime_demo'
socketio = SocketIO(app, cors_allowed_origins="*")

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

config = OmegaConf.load(args.config_path)
set_seed(config.seed)

torch.set_grad_enabled(False)

pipeline = CausalInferencePipeline(config, device=gpu)

if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location='cpu')
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")

    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"[Warning] Missing params when loading generator EMA: {missing[:8]} ...")
        if unexpected:
            print(f"[Warning] Unexpected params when loading generator EMA: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

pipeline.is_lora_enabled = False
if getattr(config, 'adapter', None) and configure_lora_for_model is not None:
    print(f"LoRA enabled with config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )

    lora_ckpt_path = getattr(config, 'lora_ckpt', None)
    if lora_ckpt_path:
        print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location='cpu')
        if isinstance(lora_checkpoint, dict) and 'generator_lora' in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint['generator_lora'])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        print("LoRA weights loaded for generator")
    else:
        print("No LoRA checkpoint specified; using base weights with initialized adapters")

    pipeline.is_lora_enabled = True

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)

# Runtime state
frame_send_queue: queue.Queue = queue.Queue()
sender_thread: Optional[Thread] = None
generation_active = False
stop_event = Event()
fp8_applied = False
torch_compile_applied = False


def generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tensor_to_base64_frame(frame_tensor: torch.Tensor) -> str:
    if frame_tensor.ndim == 3 and frame_tensor.shape[0] in (1, 3):
        frame = frame_tensor
        if frame.shape[0] == 1:
            frame = frame.repeat(3, 1, 1)
        frame = frame.permute(1, 2, 0).contiguous().cpu().numpy()
    else:
        frame = frame_tensor.cpu().numpy()
    image = Image.fromarray(frame.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def frame_sender_worker():
    global frame_send_queue, generation_active, stop_event
    print("📡 Frame sender thread started")
    while True:
        frame_data = None
        try:
            frame_data = frame_send_queue.get(timeout=1.0)
            if frame_data is None:
                frame_send_queue.task_done()
                break
            frame_tensor, frame_index, block_index, job_id = frame_data
            base64_frame = tensor_to_base64_frame(frame_tensor)
            socketio.emit('frame_ready', {
                'data': base64_frame,
                'frame_index': frame_index,
                'block_index': block_index,
                'job_id': job_id
            })
            frame_send_queue.task_done()
        except queue.Empty:
            if not generation_active and frame_send_queue.empty():
                break
        except Exception as exc:
            print(f"❌ Frame sender error: {exc}")
            if frame_data is not None:
                try:
                    frame_send_queue.task_done()
                except Exception:
                    pass
            break
    print("📡 Frame sender thread stopped")


def emit_progress(message: str, progress: int, job_id: str) -> None:
    try:
        socketio.emit('progress', {
            'message': message,
            'progress': progress,
            'job_id': job_id
        })
    except Exception as exc:
        print(f"❌ Failed to emit progress: {exc}")


@torch.no_grad()
def run_realtime_inference(prompt: str, seed: int, enable_torch_compile: bool, enable_fp8: bool, use_taehv: bool):
    global generation_active, stop_event, sender_thread, fp8_applied, torch_compile_applied

    job_id = generate_timestamp()

    try:
        generation_active = True
        stop_event.clear()

        emit_progress('Encoding prompt...', 5, job_id)

        if sender_thread is None or not sender_thread.is_alive():
            sender_thread = Thread(target=frame_sender_worker, daemon=True)
            sender_thread.start()

        prompts = [prompt] * config.num_samples
        conditional_dict = pipeline.text_encoder(text_prompts=prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                pipeline.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        if enable_torch_compile and not torch_compile_applied:
            emit_progress('Torch.compile not enabled in this demo, continuing without it.', 7, job_id)
            torch_compile_applied = False

        if enable_fp8 and not fp8_applied:
            emit_progress('FP8 quantization not supported in this build.', 8, job_id)
            fp8_applied = False

        if use_taehv:
            emit_progress('TAEHV decoder not available; using default VAE.', 9, job_id)

        emit_progress('Initializing caches...', 12, job_id)
        num_samples = config.num_samples
        num_frames = config.num_output_frames

        generator = torch.Generator(device=gpu)
        generator.manual_seed(seed)

        noise = torch.randn(
            [num_samples, num_frames, 16, 60, 104],
            device=gpu,
            dtype=torch.bfloat16,
            generator=generator
        )

        pipeline._initialize_kv_cache(batch_size=num_samples, dtype=noise.dtype, device=gpu)
        pipeline._initialize_crossattn_cache(batch_size=num_samples, dtype=noise.dtype, device=gpu)
        pipeline.generator.model.local_attn_size = pipeline.local_attn_size
        pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)

        num_blocks = num_frames // pipeline.num_frame_per_block
        all_num_frames = [pipeline.num_frame_per_block] * num_blocks
        current_start_frame = 0
        total_frames_sent = 0
        generation_start = time.time()

        emit_progress('Generating frames...', 15, job_id)

        for block_idx, current_num_frames in enumerate(all_num_frames):
            if stop_event.is_set():
                break

            progress = 15 + int(((block_idx + 1) / len(all_num_frames)) * 70)
            emit_progress(f'Processing block {block_idx + 1}/{len(all_num_frames)}', progress, job_id)

            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]
            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                if stop_event.is_set():
                    break
                timestep = torch.ones(
                    [num_samples, current_num_frames],
                    device=gpu,
                    dtype=torch.int64
                ) * current_timestep

                if index < len(pipeline.denoising_step_list) - 1:
                    _, denoised_pred = pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([
                            num_samples * current_num_frames
                        ], device=gpu, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = pipeline.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )
            if stop_event.is_set():
                break

            context_timestep = torch.ones_like(timestep) * config.context_noise
            pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )

            pixels = pipeline.vae.decode_to_pixel(denoised_pred.to(gpu), use_cache=False)
            pixels = (pixels * 0.5 + 0.5).clamp(0, 1)
            pixels = (pixels * 255).to(torch.uint8).cpu()

            block_frames = pixels.shape[1]
            for frame_offset in range(block_frames):
                if stop_event.is_set():
                    break
                frame_tensor = pixels[0, frame_offset]
                frame_send_queue.put((frame_tensor, total_frames_sent, block_idx, job_id))
                total_frames_sent += 1

            current_start_frame += current_num_frames

        if stop_event.is_set():
            emit_progress('Generation stopped.', 0, job_id)
            return

        emit_progress('Waiting for frames to finish streaming...', 95, job_id)
        frame_send_queue.join()

        generation_time = time.time() - generation_start
        emit_progress('Generation complete!', 100, job_id)
        socketio.emit('generation_complete', {
            'message': 'Video generation completed!',
            'total_frames': total_frames_sent,
            'generation_time': f"{generation_time:.2f}s",
            'job_id': job_id
        })

    except Exception as exc:
        print(f"❌ Generation failed: {exc}")
        socketio.emit('error', {
            'message': f'Generation failed: {exc}',
            'job_id': job_id
        })
    finally:
        generation_active = False
        stop_event.set()
        try:
            frame_send_queue.put(None)
        except Exception:
            pass


@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to diadistill demo server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('start_generation')
def handle_start_generation(data):
    global generation_active
    if generation_active:
        emit('error', {'message': 'Generation already in progress'})
        return

    prompt = data.get('prompt', '').strip()
    seed = int(data.get('seed', 0) or 0)
    enable_torch_compile = data.get('enable_torch_compile', False)
    enable_fp8 = data.get('enable_fp8', False)
    use_taehv = data.get('use_taehv', False)

    if not prompt:
        emit('error', {'message': 'Prompt is required'})
        return

    socketio.start_background_task(
        run_realtime_inference,
        prompt,
        seed,
        enable_torch_compile,
        enable_fp8,
        use_taehv,
    )
    emit('status', {'message': 'Generation started - frames will stream shortly'})


@socketio.on('stop_generation')
def handle_stop_generation():
    global generation_active
    generation_active = False
    stop_event.set()
    frame_send_queue.put(None)
    emit('status', {'message': 'Generation stopped'})


@app.route('/')
def index():
    return render_template('demo_fancy.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'generation_active': generation_active,
        'free_vram_gb': get_cuda_free_memory_gb(gpu),
        'fp8_applied': fp8_applied,
        'torch_compile_applied': torch_compile_applied,
    })


if __name__ == '__main__':
    print(f"🚀 Starting diadistill demo on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
