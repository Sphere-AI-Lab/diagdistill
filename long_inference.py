import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from diffusers.utils import export_to_video
# 
import numpy as np
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed
from PIL import Image

# import debugpy

# if int(os.environ.get("RANK", 0)) == 0:  # 仅主进程调试
#     debugpy.listen(("0.0.0.0", 10091))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()  # 阻塞直到调试器连接

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
parser.add_argument("--full_budget", default=False, help="Output folder")
args = parser.parse_args()

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

# inference_config

# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if 'generator_ema' in state_dict.keys():
        new_state_dict = {}
        for key, value in state_dict['generator_ema'].items():
            new_key = key.replace('._fsdp_wrapped_module', '')
            new_state_dict[new_key] = value
        state_dict['generator_ema'] = new_state_dict
        
        pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])
    else:
        pipeline.generator.load_state_dict(state_dict['generator'])

pipeline = pipeline.to(device=device, dtype=torch.bfloat16)

# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

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
# 


from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

output_folder = os.path.join(
    args.output_folder,  # 原始输出目录
    f"videos_{current_time[:]}",  # 按天分类（如videos_20240515）
)

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    if i == 0:
        #break
        continue  # Skip the first batch for testing
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    # prompt = "running"
    # prompt = "A stunningly detailed digital painting of an adorable blue cartoon kitten with big round black eyes and soft pink blush on its cheeks, sitting in a whimsical enchanted forest at night. The kitten has a fluffy round face and a curious expression, surrounded by vibrant oversized mushrooms with orange caps and white spots. Lush green trees with glowing leaves on the right, mystical purple-barked trees with bioluminescent foliage on the left. Deep blue starry night sky filled with twinkling stars and floating magical light orbs. Fantasy landscape with glowing mushrooms, sparkling fireflies, and ethereal mist. Dreamlike atmosphere with soft bokeh effects, rich colors and intricate details."
    # prompt = "A muscular young man sprinting at full speed on a rain-soaked city street, captured in cinematic slow motion. His athletic wear ripples violently in the gale-force wind, hair streaming backward from velocity, face contorted in absolute focus. The left foot impacts a pavement puddle creating a radial splash explosion. Neon-lit buildings blur into bokeh backgrounds with rain streaks forming light trails, ultra-high-speed photography frozen at 1/1000s shutter speed."
    args.i2v = False
    if args.i2v:
        # For image-to-video, batch contains image and caption
        # prompt = batch['prompts'][0]  # Get caption from batch
        prompts = [prompt] * args.num_samples

        # Process the image
        # /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/example_input_image.jpg
        # image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        image_path = '/mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/example_input_image.jpg'

        # Load the image using PIL
        image = Image.open(image_path)

        # Define transformations including resize to 480x832
        transform = transforms.Compose([
            transforms.Resize((480, 832)),  # Resize to target dimensions
            transforms.ToTensor(),          # Convert to tensor [0,1] range
            # Add any other needed transformations here
        ])

        # Apply transformations
        image_tensor = transform(image)

        # Process as in your original snippet with the reshaping
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.squeeze(0).unsqueeze(0).unsqueeze(2)  # Original processing
        image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)[:,:3,] * 2 - 1



        # Encode the input image as the first latent
        # initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        initial_latent = pipeline.vae.encode_to_latent(image_tensor).to(device=device, dtype=torch.bfloat16)
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        # prompt = "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."
        extended_prompt = None

        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples

        initial_latent = None
    
    for seed in range(10, 11):
        # seed = 40
        start_latents = None
        all_video = []
        num_overlap_frames = 3
        num_rollout = config.get('num_rollout', 10)
#改变视频的长度

        # Record start time
        full_budget = args.full_budget
        start_time = datetime.now()
        for rollout_index in range(num_rollout):

            torch.manual_seed(seed)  

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
            )
            
            # Generate 81 frames

            # Run the inference
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=start_latents,
                num_overlap_frames=num_overlap_frames,
                full_budget=full_budget,
            )

            start_frame = encode(pipeline.vae, (
                video[:, -4 * (num_overlap_frames - 1) - 1:-4 * (num_overlap_frames - 1), :] * 2.0 - 1.0
            ).transpose(2, 1).to(torch.bfloat16)).transpose(2, 1).to(torch.bfloat16)


            start_latents = torch.cat(
                [start_frame, latents[:, -(num_overlap_frames - 1):]], dim=1
            )


            # start_latents = latents[:, -(num_overlap_frames):]


            current_video = video[0].permute(0, 2, 3, 1).cpu().numpy()
            if rollout_index == 0:
                all_video.append(current_video)
            else:
                all_video.append(current_video[(4 * (num_overlap_frames - 1) + 1):])


            # all_video.append(current_video[:-(4 * (num_overlap_frames - 1) + 1)])
        
        video = np.concatenate(all_video, axis=0)
            # Record end time
        end_time = datetime.now()

        # Calculate duration
        duration = end_time - start_time
        print(f"Inference completed in: {duration}")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 提取 prompt 的前50个字符（去除空格和特殊字符，避免文件名问题）
        prompt_snippet =  prompts[0][:50].strip().replace(" ", "_").replace(".", "").replace(",", "") + "-" 



        # 4. 确保目录存在
        os.makedirs(output_folder, exist_ok=True)

        # 组合成新的视频文件名
        video_filename = "long_video_output_" + f"{prompt_snippet}_{current_time}.mp4"
        video_path = os.path.join(output_folder, video_filename)

        
        export_to_video(
            video, os.path.join(output_folder,  video_path.split('/')[-1]), fps=16)
        print(os.path.join(output_folder,  video_path.split('/')[-1]))

            # # Calculate duration
            # duration = end_time - start_time
            # print(f"Inference completed in: {duration}")
            
            # current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
            # # all_video.append(current_video)

            # all_video = [current_video]

            # num_generated_frames += latents.shape[1]

            # # Final output video
            # video = 255.0 * torch.cat(all_video, dim=1)

            # # Clear VAE cache
            # pipeline.vae.model.clear_cache()

            # # Save the video if the current prompt is not a dummy prompt
            # if idx < num_prompts:
            #     model = "regular" if not args.use_ema else "ema"
            #     for seed_idx in range(args.num_samples):
            #         # All processes save their videos

            #         current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            #         # 提取 prompt 的前50个字符（去除空格和特殊字符，避免文件名问题）
            #         prompt_snippet =  prompts[0][:50].strip().replace(" ", "_").replace(".", "").replace(",", "") + "-" 

            #         # 组合成新的视频文件名
            #         video_filename = "long_video_output_" + f"{prompt_snippet}_{current_time}.mp4"
            #         video_path = os.path.join(args.output_folder, video_filename)

            #         # export_to_video(
            #         #     video, os.path.join(args.output_folder,  video_path), fps=16)
                        
            #         if args.save_with_index:
            #             output_path = os.path.join(args.output_folder, f'{idx}-{seed}_{model}.mp4')
            #         else:
            #             output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed}-{current_time}.mp4')
            #         print(output_path)
            #         write_video(output_path, video[seed_idx], fps=16)
