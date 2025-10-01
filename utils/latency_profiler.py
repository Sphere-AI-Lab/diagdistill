import time
import torch
from datetime import datetime
from contextlib import contextmanager
import json
from typing import Dict, List, Optional
import psutil
import GPUtil


class LatencyProfiler:
    """
    A comprehensive latency profiler for video generation pipeline
    """
    def __init__(self):
        self.timings = {}
        self.gpu_memory_usage = {}
        self.system_memory_usage = {}
        self.current_context = None
        
    def reset(self):
        """Reset all timing records"""
        self.timings = {}
        self.gpu_memory_usage = {}
        self.system_memory_usage = {}
        
    @contextmanager
    def time_context(self, name: str):
        """Context manager for timing code blocks"""
        self.current_context = name
        start_time = time.perf_counter()
        
        # Record GPU memory before
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
            
        # Record system memory before
        system_memory_before = psutil.virtual_memory().used / 1024**3  # GB
        
        try:
            yield
        finally:
            # Ensure GPU operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Record timing
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)
            
            # Record GPU memory after
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_used = gpu_memory_after - gpu_memory_before
                if name not in self.gpu_memory_usage:
                    self.gpu_memory_usage[name] = []
                self.gpu_memory_usage[name].append({
                    'before': gpu_memory_before,
                    'after': gpu_memory_after,
                    'used': gpu_memory_used
                })
            
            # Record system memory after
            system_memory_after = psutil.virtual_memory().used / 1024**3  # GB
            system_memory_used = system_memory_after - system_memory_before
            if name not in self.system_memory_usage:
                self.system_memory_usage[name] = []
            self.system_memory_usage[name].append({
                'before': system_memory_before,
                'after': system_memory_after,
                'used': system_memory_used
            })
            
            print(f"[{name}] Time: {duration:.3f}s, GPU Memory: {gpu_memory_used:.2f}GB")
            
            self.current_context = None
    
    def get_summary(self) -> Dict:
        """Get summary of all timing measurements"""
        summary = {
            'timings': {},
            'gpu_memory': {},
            'system_memory': {},
            'total_time': 0
        }
        
        total_time = 0
        for name, times in self.timings.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_component_time = sum(times)
            
            summary['timings'][name] = {
                'count': len(times),
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'total': total_component_time,
                'all_times': times
            }
            total_time += total_component_time
            
        summary['total_time'] = total_time
        
        # GPU memory summary
        for name, memories in self.gpu_memory_usage.items():
            if memories:
                avg_used = sum(m['used'] for m in memories) / len(memories)
                max_used = max(m['used'] for m in memories)
                summary['gpu_memory'][name] = {
                    'avg_used': avg_used,
                    'max_used': max_used,
                    'count': len(memories)
                }
        
        # System memory summary  
        for name, memories in self.system_memory_usage.items():
            if memories:
                avg_used = sum(m['used'] for m in memories) / len(memories)
                max_used = max(m['used'] for m in memories)
                summary['system_memory'][name] = {
                    'avg_used': avg_used,
                    'max_used': max_used,
                    'count': len(memories)
                }
                
        return summary
    
    def print_summary(self):
        """Print detailed summary of measurements"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("VIDEO GENERATION LATENCY PROFILING REPORT")
        print("="*60)
        
        print(f"\nTOTAL EXECUTION TIME: {summary['total_time']:.3f}s")
        
        print("\nCOMPONENT BREAKDOWN:")
        print("-" * 60)
        print(f"{'Component':<25} {'Count':<6} {'Avg(s)':<8} {'Min(s)':<8} {'Max(s)':<8} {'Total(s)':<8}")
        print("-" * 60)
        
        for name, timing in summary['timings'].items():
            print(f"{name:<25} {timing['count']:<6} {timing['avg']:<8.3f} "
                  f"{timing['min']:<8.3f} {timing['max']:<8.3f} {timing['total']:<8.3f}")
        
        if summary['gpu_memory']:
            print("\nGPU MEMORY USAGE:")
            print("-" * 40)
            print(f"{'Component':<25} {'Avg Used(GB)':<12} {'Max Used(GB)':<12}")
            print("-" * 40)
            
            for name, memory in summary['gpu_memory'].items():
                print(f"{name:<25} {memory['avg_used']:<12.2f} {memory['max_used']:<12.2f}")
        
        # Show current GPU utilization
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assume using first GPU
                    print(f"\nCURRENT GPU STATUS:")
                    print(f"GPU Utilization: {gpu.load*100:.1f}%")
                    print(f"Memory Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
                    print(f"Temperature: {gpu.temperature}°C")
            except:
                pass
                
        print("="*60)
    
    def save_report(self, filepath: str):
        """Save detailed report to JSON file"""
        summary = self.get_summary()
        summary['timestamp'] = datetime.now().isoformat()
        
        # Add system info
        summary['system_info'] = {
            'python_version': torch.__version__,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            summary['system_info']['cuda_version'] = torch.version.cuda
            summary['system_info']['gpu_name'] = torch.cuda.get_device_name(0)
            summary['system_info']['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Detailed report saved to: {filepath}")


# Global profiler instance
profiler = LatencyProfiler()


def measure_latency_comprehensive(
    pipeline, 
    prompts: List[str], 
    sampled_noise: torch.Tensor,
    initial_latent: Optional[torch.Tensor] = None,
    num_runs: int = 1
):
    """
    Comprehensive latency measurement for video generation pipeline
    """
    profiler.reset()
    
    print(f"Starting comprehensive latency measurement for {num_runs} runs...")
    
    all_results = []
    
    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        
        with profiler.time_context("total_inference"):
            
            # Text encoding
            with profiler.time_context("text_encoding"):
                text_embeddings = pipeline.text_encoder.encode_text(prompts)
            
            # Pipeline inference (this includes all denoising steps)
            with profiler.time_context("pipeline_inference"):
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    return_latents=True,
                    initial_latent=initial_latent,
                )
            
            # VAE decoding (if not already included in pipeline)
            with profiler.time_context("vae_decoding"):
                # This might already be done in pipeline.inference
                # but we can measure it separately if needed
                pass
                
            # Video post-processing
            with profiler.time_context("video_postprocessing"):
                from einops import rearrange
                processed_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
                
        all_results.append({
            'video': processed_video,
            'latents': latents
        })
    
    profiler.print_summary()
    
    return all_results, profiler.get_summary()


# Decorator for easy function timing
def time_function(name: str = None):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = name or f"{func.__module__}.{func.__name__}"
            with profiler.time_context(func_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator