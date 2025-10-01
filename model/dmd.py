from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel, SpatialHead, TemporalHead, UpsamplingModule

from typing import Optional, Tuple, Union
import random
import torch.nn as nn
import yaml
import argparse
from pathlib import Path




@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()


        self.pred_spatial_head = SpatialHead(num_channels=16, num_layers=3).to_empty(device="cuda")

        self.target_spatial_head = SpatialHead(num_channels=16, num_layers=3).to_empty(device="cuda")
        self.target_spatial_head.requires_grad_(False)
        
        # EMA decay rate for spatial heads
        self.spatial_head_ema_decay = getattr(args, "spatial_head_ema_decay", 0.95)


        def init_spatial_head(spatial_head):
        
            for layer in spatial_head.layers:
                conv = layer[0] 
                nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.zeros_(conv.bias)
            
           
            for layer in spatial_head.layers:
                groupnorm = layer[1]  #
                nn.init.ones_(groupnorm.weight)
                nn.init.zeros_(groupnorm.bias)
            
        
            nn.init.kaiming_normal_(spatial_head.conv_out.weight, mode='fan_in')
            nn.init.zeros_(spatial_head.conv_out.bias)

      
        init_spatial_head(self.pred_spatial_head)

        init_spatial_head(self.target_spatial_head)

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True,
        pred_spatial_head=None,
        target_spatial_head=None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )
        # 
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)


   

        cd_target = 'learn'
        cd_target = "learn"
        pred_fake_image_cd = self.prepare_cd_target(pred_fake_image.transpose(1,2), cd_target)
        if cd_target in ["learn", "hlearn"]:
            pred_fake_image_cd = pred_spatial_head(pred_fake_image_cd)
       
        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dtype):
                pred_real_image_cd = self.prepare_cd_target(pred_real_image.transpose(1,2), cd_target)
                if cd_target in ["learn", "hlearn"]:
                    pred_real_image_cd = target_spatial_head(pred_real_image_cd)

        # if args.loss_type == "l2":

        def dynamic_frame_weights(pred, target):
  
            per_frame_loss = F.mse_loss(
                pred, 
                target, 
                reduction='none'
            ).mean(dim=[1, 3, 4])  
            
         
            cum_error = torch.cumsum(per_frame_loss, dim=1)  
            
          
            weights = 1.0 + cum_error / cum_error[:, -1].unsqueeze(1) 
            
            return weights.detach()  

        dyn_weights = dynamic_frame_weights(pred_fake_image_cd, pred_real_image_cd)


        def exponential_weights(num_frames: int) -> torch.Tensor:
            """生成指数递增权重（支持GPU自动转移）
            Args:
                num_frames: 视频帧数
            Returns:
                weights: [num_frames] 的权重张量
            """
            exp_base = 1.2
            frame_idx = torch.arange(num_frames)
            weights = torch.pow(exp_base, frame_idx)  
            return weights / weights.mean()  

        exp_weights = exponential_weights(pred_fake_image_cd.shape[2])  # [T]
        exp_weights = exp_weights.view(1, -1).expand_as(dyn_weights).to(dyn_weights)  # [B,T]

        hybrid_weights = None
        hybrid_weights = 0.7 * dyn_weights + 0.3 * exp_weights


        grad_motion = (pred_fake_image_cd - pred_real_image_cd)

        # flow_loss = (hybrid_weights * per_frame_grad).mean()
        # if hybrid_weight is not None:

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer

            # motion_grad
            with torch.no_grad():
                with torch.autocast("cuda", dtype=self.dtype):
                    estimated_clean_image_or_video_cd = self.prepare_cd_target(estimated_clean_image_or_video.transpose(1,2), cd_target)
                    if cd_target in ["learn", "hlearn"]:
                        estimated_clean_image_or_video_cd = self.target_spatial_head(estimated_clean_image_or_video_cd)
            p_real_motion = (estimated_clean_image_or_video_cd - pred_real_image_cd)
            normalizer_motion = torch.abs(p_real_motion).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad_motion = grad_motion / normalizer_motion



        grad = torch.nan_to_num(grad)
        grad_motion = torch.nan_to_num(grad_motion)

        return grad, grad_motion, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "dmdtrain_gradient_motion_norm": torch.mean(torch.abs(grad_motion)).detach(),
            "timestep": timestep.detach()
        }


    def prepare_cd_target(self, latent, target: str, spatial_head: Optional[nn.Module] = None):
        # latent shape: b, c, t, h, w
        b, c, t, h, w = latent.shape

        if target in ["raw", "learn", "hlearn"]:
            return latent
        # elif target == "learn":
        #     latent = spatial_head(latent)
        #     return latent
        elif target == "diff":
            latent_prev = latent[:, :, :-1]
            latent_next = latent[:, :, 1:]
            diff = latent_next - latent_prev
            return diff
        elif target in ["freql", "freqh"]:
            shape = latent.shape
            shape = (1, *shape[1:])
            low_pass_filter = get_free_init_freq_filter(shape, latent.device)
            if target == "freql":
                return apply_freq_filter(latent, low_pass_filter, out_freq="low")
            elif target == "freqh":
                return apply_freq_filter(latent, low_pass_filter, out_freq="high")
            else:
                raise ValueError(f"Invalid target: {target}")
        elif target in ["lcor", "gcor", "sgcor", "sgcord"]:
            latent_prev = latent[:, :, :-1]
            latent_next = latent[:, :, 1:]
            latent_prev = rearrange(latent_prev, "b c t h w -> (b t) c h w")
            latent_next = rearrange(latent_next, "b c t h w -> (b t) c h w")

            if target == "lcor":
                flow, _, corr = local_correlation_softmax(latent_prev, latent_next, 7)
                return corr
            elif target in ["gcor", "sgcor", "sgcord"]:
                flow, _, corr = global_correlation_softmax(latent_prev, latent_next)
                if target == "gcor":
                    return corr
                elif target == "sgcor":
                    return corr * (h**0.5)
                elif target == "sgcord":
                    raise NotImplementedError("Not implemented yet")
        else:
            raise ValueError(f"Invalid target: {target}")



    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # EMA update instead of direct state dict copy
            update_ema(
                self.target_spatial_head.parameters(),
                self.pred_spatial_head.parameters(),
                rate=self.spatial_head_ema_decay
            )

            # Step 2: Compute the KL grad
            grad, grad_motion, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                pred_spatial_head = self.pred_spatial_head,
                target_spatial_head = self.target_spatial_head,   
            )

        cd_target = "learn"
        with torch.no_grad():
            with torch.autocast("cuda", dtype=self.dtype):
                original_latent_cd = self.prepare_cd_target(original_latent.transpose(1,2), cd_target)
                if cd_target in ["learn", "hlearn"]:
                    original_latent_cd = self.target_spatial_head(original_latent_cd)

        # 


        def dynamic_frame_weights(pred, target):
      
            per_frame_loss = F.mse_loss(
                pred, 
                target, 
                reduction='none'
            ).mean(dim=[1, 3, 4])  
            
          
            cum_error = torch.cumsum(per_frame_loss, dim=1)  # [2, 21]
          
            weights = 1.0 + cum_error / cum_error[:, -1].unsqueeze(1)  
            
            return weights.detach() 

        dyn_weights = dynamic_frame_weights(grad_motion, original_latent_cd)


        def exponential_weights(num_frames: int) -> torch.Tensor:
            """生成指数递增权重（支持GPU自动转移）
            Args:
                num_frames: 视频帧数
            Returns:
                weights: [num_frames] 的权重张量
            """
            exp_base = 1.2
            frame_idx = torch.arange(num_frames)
            weights = torch.pow(exp_base, frame_idx)  
            return weights / weights.mean() 

        exp_weights = exponential_weights(grad_motion.shape[2])  # [T]
        exp_weights = exp_weights.view(1, -1).expand_as(dyn_weights).to(dyn_weights)  # [B,T]

        hybrid_weights = 0.7 * dyn_weights + 0.3 * exp_weights


        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        
            diff = (original_latent_cd.double() - grad_motion.double()).detach()  # [B, C, T, H, W]

        
            squared_error = (original_latent_cd.double() - diff) ** 2  # [B, C, T, H, W]
            weighted_squared_error = squared_error * hybrid_weights.view(1, 1, -1, 1, 1) 

     
            dmd_motion_loss = weighted_squared_error.mean() 
            dmd_original_loss = dmd_loss
            if self.args.use_dmd_loss:
                dmd_loss = dmd_loss + 1.0 * dmd_motion_loss



       

        return dmd_loss, dmd_motion_loss, dmd_original_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        current_step=None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            current_step=current_step,
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_motion_loss, dmd_original_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        return dmd_loss, dmd_motion_loss, dmd_original_loss, dmd_log_dict, pred_image

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        min_timestep = self.min_score_timestep
        max_timestep = self.num_train_timestep
        
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
