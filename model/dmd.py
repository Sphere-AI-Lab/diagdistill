# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import time
import copy

from model.base import SelfForcingModel
from model.spatial_head import SpatialHead
from utils.memory import log_gpu_memory
import torch.distributed as dist
from utils.debug_option import DEBUG, LOG_GPU_MEMORY


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
        self.min_num_training_frames = getattr(args, "min_num_training_frames", 21)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

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

        # Optional motion feature regression branch: latent diff -> SpatialHead.
        self.use_flow_reg_loss = getattr(args, "use_flow_reg_loss", False)
        self.flow_reg_ema_decay = float(getattr(args, "flow_reg_ema_decay", 0.95))
        self.motion_head_hidden_dim = int(getattr(args, "motion_head_hidden_dim", 64))
        self.motion_head_num_layers = int(getattr(args, "motion_head_num_layers", 2))
        self.motion_head_kernel_size = int(getattr(args, "motion_head_kernel_size", 1))
        self.lambda_spatial_dmd = float(getattr(args, "lambda_spatial_dmd", 1.0))
        self.lambda_flow_dmd = float(getattr(args, "lambda_flow_dmd", 1.0))
        self.gamma_temporal = float(getattr(args, "gamma_temporal", 1.0))
        self.lambda_reg = float(getattr(args, "lambda_reg", 0.0))
        self.reg_loss_type = str(getattr(args, "reg_loss_type", "mse")).lower()
        self.reg_loss_eps = float(getattr(args, "reg_loss_eps", 1e-3))
        self.reg_loss_cauchy_c = float(getattr(args, "reg_loss_cauchy_c", 1e-2))
        self.use_teacher_4step_regression = bool(getattr(args, "use_teacher_4step_regression", False))
        self.teacher_4step_list = list(getattr(args, "teacher_4step_list", [1000, 750, 500, 250]))

        valid_reg_loss_types = {"mse", "charbonnier", "cauchy"}
        if self.reg_loss_type not in valid_reg_loss_types:
            raise ValueError(
                f"Invalid reg_loss_type '{self.reg_loss_type}'. "
                f"Supported: {sorted(valid_reg_loss_types)}"
            )

        self.motion_head_student = None
        self.motion_head_teacher = None
        self.regression_teacher_generator = None
        if self.use_flow_reg_loss:
            num_channels = getattr(args, "image_or_video_shape", [1, 1, 16, 60, 104])[2]
            self.motion_head_student = SpatialHead(
                num_channels=num_channels,
                num_layers=self.motion_head_num_layers,
                kernel_size=self.motion_head_kernel_size,
                hidden_dim=self.motion_head_hidden_dim,
            ).to(device=device, dtype=self.dtype)
            self.motion_head_teacher = SpatialHead(
                num_channels=num_channels,
                num_layers=self.motion_head_num_layers,
                kernel_size=self.motion_head_kernel_size,
                hidden_dim=self.motion_head_hidden_dim,
            ).to(device=device, dtype=self.dtype)
            self.motion_head_teacher.load_state_dict(self.motion_head_student.state_dict())
            self.motion_head_teacher.requires_grad_(False)

    @torch.no_grad()
    def freeze_regression_teacher_from_generator(self):
        if not self.use_teacher_4step_regression:
            return
        if self.regression_teacher_generator is not None:
            return
        self.regression_teacher_generator = copy.deepcopy(self.generator)
        # Keep the frozen teacher fully on the current rank device to avoid cpu/cuda mismatches in rollout.
        self.regression_teacher_generator = self.regression_teacher_generator.to(self.device)
        self.regression_teacher_generator.requires_grad_(False)
        self.regression_teacher_generator.eval()

    def _resolve_step_list(self, step_list):
        steps = torch.tensor(step_list, dtype=torch.long)
        if getattr(self.args, "warp_denoising_step", False):
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            steps = timesteps[1000 - steps]
        return steps.to(self.device)

    def _run_generator_with_fixed_teacher_steps(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
        slice_last_frames: int = 21,
    ):
        teacher_generator = self.regression_teacher_generator
        if teacher_generator is None:
            teacher_generator = self.generator

        original_generator = self.generator
        original_pipeline = self.inference_pipeline
        original_steps = self.denoising_step_list
        original_last_step_only = getattr(self.args, "last_step_only", False)
        try:
            self.generator = teacher_generator
            # Teacher regression target should always come from the final step
            # of the fixed 4-step trajectory instead of a randomly early-exited step.
            self.args.last_step_only = True
            self.inference_pipeline = None
            self.denoising_step_list = self._resolve_step_list(self.teacher_4step_list)
            return self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames,
            )
        finally:
            self.generator = original_generator
            self.inference_pipeline = original_pipeline
            self.denoising_step_list = original_steps
            self.args.last_step_only = original_last_step_only

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
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

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # Motion gradient on temporal differences.
        if pred_fake_image.shape[1] > 1:
            grad_motion = (
                (pred_fake_image[:, 1:] - pred_fake_image[:, :-1]) -
                (pred_real_image[:, 1:] - pred_real_image[:, :-1])
            )
        else:
            grad_motion = grad[:, :0]

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
            if pred_real_image.shape[1] > 1:
                p_real_motion = (
                    (estimated_clean_image_or_video[:, 1:] - estimated_clean_image_or_video[:, :-1]) -
                    (pred_real_image[:, 1:] - pred_real_image[:, :-1])
                )
                normalizer_motion = torch.abs(p_real_motion).mean(dim=[1, 2, 3, 4], keepdim=True)
                grad_motion = grad_motion / normalizer_motion
        grad = torch.nan_to_num(grad)
        grad_motion = torch.nan_to_num(grad_motion)

        return grad, grad_motion, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "dmdtrain_gradient_motion_norm": torch.mean(torch.abs(grad_motion)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        regression_target: Optional[torch.Tensor] = None,
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

            # Step 2: Compute the KL grad
            grad, grad_motion, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        original_target = (original_latent.double() - grad.double()).detach()
        regression_target_for_reg = original_target
        if regression_target is not None:
            regression_target_for_reg = regression_target.to(dtype=torch.double).detach()
        if gradient_mask is not None:
            dmd_original_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                original_target[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_original_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                original_target,
                reduction="mean",
            )

        # Motion loss branch: mirror the original DMD target construction.
        if grad_motion.shape[1] > 0:
            def dynamic_frame_weights(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                per_frame_loss = F.mse_loss(pred, target, reduction="none").mean(dim=[2, 3, 4])
                cum_error = torch.cumsum(per_frame_loss, dim=1)
                denom = cum_error[:, -1:].clamp_min(1e-6)
                return (1.0 + cum_error / denom).detach()

            dyn_weights = dynamic_frame_weights(grad_motion, original_latent[:, 1:])

            def exponential_weights(num_frames: int) -> torch.Tensor:
                exp_base = 1.2
                frame_idx = torch.arange(num_frames, device=grad_motion.device)
                weights = torch.pow(exp_base, frame_idx)
                return weights / weights.mean()

            exp_weights = exponential_weights(grad_motion.shape[1]).view(1, -1).expand_as(dyn_weights)
            hybrid_weights = 0.7 * dyn_weights + 0.3 * exp_weights

            pred_motion = (original_latent[:, 1:] - original_latent[:, :-1]).double()
            target_motion = (pred_motion - grad_motion.double()).detach()
            squared_error = (pred_motion - target_motion) ** 2
            weighted_squared_error = squared_error * hybrid_weights.view(
                hybrid_weights.shape[0], hybrid_weights.shape[1], 1, 1, 1
            ).double()
            dmd_motion_loss = weighted_squared_error.mean()
        else:
            dmd_motion_loss = dmd_original_loss * 0.0

        use_motion_loss = getattr(
            self.args, "use_motion_loss", getattr(self.args, "use_dmd_loss", True)
        )
        flow_dmd_term = dmd_motion_loss if use_motion_loss else dmd_original_loss * 0.0

        flow_reg_loss = dmd_original_loss * 0.0
        if (
            self.use_flow_reg_loss
            and self.motion_head_student is not None
            and self.motion_head_teacher is not None
            and original_latent.shape[1] > 1
        ):
            delta_student = (original_latent[:, 1:] - original_latent[:, :-1]).to(self.dtype)
            delta_teacher = (
                regression_target_for_reg[:, 1:] - regression_target_for_reg[:, :-1]
            ).to(self.dtype)
            pred_motion_feature = self.motion_head_student(delta_student)
            with torch.no_grad():
                target_motion_feature = self.motion_head_teacher(delta_teacher)

            if gradient_mask is not None:
                pair_mask = gradient_mask[:, 1:] & gradient_mask[:, :-1]
                if pair_mask.any():
                    flow_reg_loss = F.mse_loss(
                        pred_motion_feature[pair_mask],
                        target_motion_feature[pair_mask],
                        reduction="mean",
                    )
                else:
                    flow_reg_loss = dmd_original_loss * 0.0
            else:
                flow_reg_loss = F.mse_loss(
                    pred_motion_feature,
                    target_motion_feature,
                    reduction="mean",
                )

        # L_total = lambda_spatial * L_spatial_DMD + lambda_reg * L_reg
        #           + gamma * (lambda_flow * L_flow_DMD + L_flow_reg)
        reg_loss = self._compute_regression_loss(
            prediction=original_latent.double(),
            target=regression_target_for_reg,
            gradient_mask=gradient_mask,
        )

        dmd_loss = (
            self.lambda_spatial_dmd * dmd_original_loss
            + self.lambda_reg * reg_loss
            + self.gamma_temporal * (self.lambda_flow_dmd * flow_dmd_term + flow_reg_loss)
        )

        dmd_log_dict.update({
            "dmd_motion_loss": dmd_motion_loss.detach(),
            "dmd_original_loss": dmd_original_loss.detach(),
            "flow_reg_loss": flow_reg_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "lambda_spatial_dmd": torch.tensor(self.lambda_spatial_dmd, device=dmd_original_loss.device),
            "lambda_flow_dmd": torch.tensor(self.lambda_flow_dmd, device=dmd_original_loss.device),
            "gamma_temporal": torch.tensor(self.gamma_temporal, device=dmd_original_loss.device),
            "lambda_reg": torch.tensor(self.lambda_reg, device=dmd_original_loss.device),
            "reg_loss_type_id": torch.tensor(
                {"mse": 0, "charbonnier": 1, "cauchy": 2}[self.reg_loss_type],
                device=dmd_original_loss.device,
            ),
        })
        return dmd_loss, dmd_log_dict

    def _compute_regression_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gradient_mask is not None:
            prediction = prediction[gradient_mask]
            target = target[gradient_mask]

        if self.reg_loss_type == "mse":
            return F.mse_loss(prediction, target, reduction="mean")

        diff = prediction - target
        if self.reg_loss_type == "charbonnier":
            eps = max(self.reg_loss_eps, 1e-12)
            return torch.sqrt(diff * diff + eps * eps).mean()

        c = max(self.reg_loss_cauchy_c, 1e-12)
        return torch.log1p((diff / c) ** 2).mean()

    @torch.no_grad()
    def update_motion_head_teacher(self):
        if (
            not self.use_flow_reg_loss
            or self.motion_head_student is None
            or self.motion_head_teacher is None
        ):
            return

        decay = self.flow_reg_ema_decay
        for teacher_param, student_param in zip(
            self.motion_head_teacher.parameters(),
            self.motion_head_student.parameters(),
        ):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
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
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        # Step 1: Unroll generator to obtain fake videos
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        _t_gen_start = time.time()
        regression_target = None
        rng_state_cpu = None
        rng_state_cuda = None
        if self.use_teacher_4step_regression:
            rng_state_cpu = torch.get_rng_state()
            if torch.cuda.is_available():
                rng_state_cuda = torch.cuda.get_rng_state(self.device)
        if DEBUG and dist.get_rank() == 0:
            print(f"generator_rollout")
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
            slice_last_frames=slice_last_frames
        )
        if self.use_teacher_4step_regression:
            torch.set_rng_state(rng_state_cpu)
            if torch.cuda.is_available() and rng_state_cuda is not None:
                torch.cuda.set_rng_state(rng_state_cuda, self.device)
            with torch.no_grad():
                regression_target, _, _, _ = self._run_generator_with_fixed_teacher_steps(
                    image_or_video_shape=image_or_video_shape,
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent,
                    slice_last_frames=slice_last_frames,
                )
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {pred_image.shape}")
            if gradient_mask is not None:   
                print(f"gradient_mask: {gradient_mask[0, :, 0, 0, 0]}")
            else:
                print(f"gradient_mask: None")
        gen_time = time.time() - _t_gen_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After generator unroll", device=self.device, rank=dist.get_rank())
        # Step 2: Compute the DMD loss
        _t_loss_start = time.time()
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            regression_target=regression_target,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Generator loss: After compute_distribution_matching_loss", device=self.device, rank=dist.get_rank())
        try:
            loss_val = dmd_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start
        # print(f"[GeneratorLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")

        dmd_log_dict.update({
            "gen_time": gen_time,
            "loss_time": loss_time
        })

        return dmd_loss, dmd_log_dict

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
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: Before generator unroll", device=self.device, rank=dist.get_rank())
        slice_last_frames = getattr(self.args, "slice_last_frames", 21)
        # Step 1: Run generator on backward simulated noisy input
        _t_gen_start = time.time()
        with torch.no_grad():
            if DEBUG and dist.get_rank() == 0:
                print(f"critic_rollout")
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                slice_last_frames=slice_last_frames
            )
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {generated_image.shape}")
        gen_time = time.time() - _t_gen_start
        batch_size, num_frame = generated_image.shape[:2]
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After generator unroll", device=self.device, rank=dist.get_rank())
        _t_loss_start = time.time()

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            batch_size,
            num_frame,
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
        ).unflatten(0, (batch_size, num_frame))

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
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        try:
            loss_val = denoising_loss.item()
        except Exception:
            loss_val = float('nan')
        loss_time = time.time() - _t_loss_start
        if (not dist.is_initialized() or dist.get_rank() == 0) and LOG_GPU_MEMORY:
            log_gpu_memory(f"Critic loss: After denoising loss", device=self.device, rank=dist.get_rank())
        # print(f"[CriticLoss] loss {loss_val} | gen_time {gen_time:.3f}s | loss_time {loss_time:.3f}s")


        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "gen_time": gen_time,
            "loss_time": loss_time
        }

        return denoising_loss, critic_log_dict
