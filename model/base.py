from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

# 
class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
        current_step=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            current_step=current_step,
            **conditional_dict,
        )
        
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        current_step=None,
        **conditional_dict: dict,
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, 
            current_step=current_step,
            **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise
        )

import torch
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange
# 

class SpatialHead(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        assert num_layers >= 2, "num_layers must be at least 2"

        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.in_act = nn.SiLU()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        num_channels if i == 0 else hidden_dim,
                        hidden_dim,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                    ),
                    nn.GroupNorm(
                        num_groups=norm_num_groups,
                        num_channels=hidden_dim,
                        eps=norm_eps,
                    ),
                    nn.SiLU(),
                )
                for i in range(num_layers - 1)
            ]
        )

        self.conv_out = nn.Conv2d(hidden_dim, num_channels, kernel_size=1, padding=0)

        # zero initialize the last layer
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    # def forward(self, x):
    #     # x shape: b, c, t, h, w
    #     b, c, t, h, w = x.shape
    #     x_in = x
    #     x = rearrange(x, "b c t h w -> (b t) c h w").contiguous()  # 强制拷贝
    #     x = self.in_act(x)
    #     for layer in self.layers:
    #         x = layer(x)

    #     x = self.conv_out(x)
    #     x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t).contiguous()
    #     x = x + x_in  # 非 inplace 操作
    #     return x

    def forward(self, x):
        # x shape: (b, c, t, h, w)
        b, c, t, h, w = x.shape
        x_in = x  # 保存原始输入
        
        # 替代 rearrange(x, "b c t h w -> (b t) c h w")
        x_reshaped = torch.zeros(b * t, c, h, w, dtype=x.dtype, device=x.device)
        for bi in range(b):
            for ti in range(t):
                x_reshaped[bi * t + ti] = x[bi, :, ti]  # 手动展开时间维度
        
        # 正常处理卷积
        x_reshaped = self.in_act(x_reshaped)
        for layer in self.layers:
            x_reshaped = layer(x_reshaped)
        x_reshaped = self.conv_out(x_reshaped)
        
        # 替代 rearrange(x, "(b t) c h w -> b c t h w")
        x_out = torch.zeros(b, c, t, h, w, dtype=x.dtype, device=x.device)
        for bi in range(b):
            for ti in range(t):
                x_out[bi, :, ti] = x_reshaped[bi * t + ti]  # 手动恢复时间维度
        
        return x_out + x_in  # 非inplace加法

class TemporalHead(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        assert num_layers >= 2, "num_layers must be at least 2"

        super().__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.in_act = nn.SiLU()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        num_channels if i == 0 else hidden_dim,
                        hidden_dim,
                        kernel_size=self.kernel_size,
                        padding=self.padding,
                    ),
                    nn.GroupNorm(
                        num_groups=norm_num_groups,
                        num_channels=hidden_dim,
                        eps=norm_eps,
                    ),
                    nn.SiLU(),
                )
                for i in range(num_layers - 1)
            ]
        )

        self.conv_out = nn.Conv1d(hidden_dim, num_channels, kernel_size=1, padding=0)

        # zero initialize the last layer
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        # x shape: b, c, t, h, w -> b*h*w, c, t
        b, c, t, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c t h w -> (b h w) c t")
        x = self.in_act(x)
        for layer in self.layers:
            x = layer(x)

        x = self.conv_out(x)

        x = rearrange(x, "(b h w) c t -> b c t h w", b=b, h=h, w=w)
        x = x + x_in
        return x



class IdentitySpatialHead(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self):
        super().__init__()

        self.identity_layer = nn.Identity()

    def forward(self, x):
        return self.identity_layer(x)


class UpsamplingModule(nn.Module):
    """
    上采样模块实现，包含单个卷积层和PixelShuffle操作
    参数:
        in_channels: 输入特征图的通道数
        r: 上采样因子(默认=4)
    """
    def __init__(self, in_channels, out_channels, r=4):
        super(UpsamplingModule, self).__init__()
        self.r = r
        
        # 单个卷积层：增加通道数为r²倍

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,  # 输出通道数 = in_channels × r²
            kernel_size=3,
            padding=1,
            stride=1
        )

        self.conv1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * (r ** 2),  # 输出通道数 = in_channels × r²
            kernel_size=3,
            padding=1,
            stride=1
        )


        
        # PixelShuffle操作：将通道数转换为空间分辨率
        self.pixel_shuffle = nn.PixelShuffle(r)
        
        # 可选：初始化卷积层权重
        nn.init.kaiming_normal_(self.conv0.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv0.bias, 0)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        """
        前向传播
        输入x形状: (batch_size, in_channels, H, W)
        输出形状: (batch_size, in_channels, H*r, W*r)
        """
        # 1. 通过卷积增加通道数
        x = self.conv0(x)

        x = self.conv1(x)  # 输出形状: (batch, in_channels*r², H, W)
        
        # 2. 应用PixelShuffle进行上采样
        x = self.pixel_shuffle(x)  # 输出形状: (batch, in_channels, H*r, W*r)
        
        return x

# 使用示例
# if __name__ == "__main__":
#     # 创建模块实例
#     upsampler = UpsamplingModule(in_channels=16, out_channels=3, r=4)
    
#     # 创建随机输入 (batch=2, channels=64, height=32, width=32)
#     input_tensor = torch.randn(2, 16, 32, 32)
    
#     # 前向传播
#     output = upsampler(input_tensor)
    
#     print(f"输入形状: {input_tensor.shape}")
#     print(f"输出形状: {output.shape}")
#     # 预期输出: torch.Size([2, 64, 128, 128])

if __name__ == "__main__":
    head = SpatialHead(num_channels=4, num_layers=3)
    # head = TemporalHead(num_channels=4, num_layers=3)
    x = torch.randn(2, 4, 3, 64, 64)
    out = head(x)
    diff = out - x
    # print(diff.abs().max())
    print(out.shape)

