from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import yaml
from pathlib import Path
import argparse


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        # self.independent_first_frame = True
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block


    def generate_timestep_matrix_DD(
        self,
        num_frames,
        step_template,
        base_num_frames,
        ar_step=5,
        num_pre_ready=0,
        casual_block_size=1,
        constrain_mask=False,
        shrink_interval_with_mask=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple]]:
        step_matrix, step_index = [], []
        update_mask, valid_interval = [], []
        # num_iterations = len(step_template) + 1
        num_frames_block = num_frames // casual_block_size
        base_num_frames_block = base_num_frames // casual_block_size
        # TODO: Test
        num_iterations = base_num_frames_block 
        if base_num_frames_block < num_frames_block:
            infer_step_num = len(step_template)
            gen_block = base_num_frames_block
            min_ar_step = infer_step_num / gen_block
            assert ar_step >= min_ar_step, f"ar_step should be at least {math.ceil(min_ar_step)} in your setting"
        # print(num_frames, step_template, base_num_frames, ar_step, num_pre_ready, casual_block_size, num_frames_block, base_num_frames_block)
        step_template = torch.cat(
            [
                torch.tensor([999], dtype=torch.int64, device=step_template.device),
                step_template.long(),
                torch.tensor([0], dtype=torch.int64, device=step_template.device),
            ]
        )  # to handle the counter in row works starting from 1
        pre_row = torch.zeros(num_frames_block, dtype=torch.long)
        if num_pre_ready > 0:
            pre_row[: num_pre_ready // casual_block_size] = num_iterations

        device = step_template.device
        step_type = "random"
        step_type = "diagonal"

        if step_type == "diagonal":
            # index, indices = torch.sort(index, dim=1, descending=True)     
            # step_num_type = self.step_num_type # can be a random value (4 - block) 1. 4 尽可能多 2. 尽快到 0
            step_num_type = 4
            start_step = 999 
            end_step = 499
            denoising_step_list = []
            # 
            # 
            # for step_num in range(step_num_type):
            #     step_list = list(torch.linspace(start_step, end_step, step_num+1).round().long())    
            #     denoising_step_list.append(step_list) 
            
            # for step_num in range(step_num_type):
            #     step_list = 
            #     denoising_step_list.append()
            step_list = [tensor for tensor in self.denoising_step_list]
            try:
                denoising_step_list = [[step_list[0], step_list[-1]], [step_list[0],step_list[1], step_list[-1]], step_list]
                denoising_step_list = denoising_step_list[::-1]
            except:
                denoising_step_list = self.denoising_step_list

            

            for list_idx in range(len(denoising_step_list)):
                
                new = torch.stack(denoising_step_list[list_idx]).to(step_template)
                new = torch.concat([torch.tensor([999]).to(new), new])
                new = torch.concat([new, torch.tensor([0]).to(new)])

                denoising_step_list[list_idx] = new

            timesteps_list = denoising_step_list

        while torch.all(pre_row >= (num_iterations - 1)) == False:
            new_row = torch.zeros(num_frames_block, dtype=torch.long)
            for i in range(num_frames_block):
                if i == 0 or pre_row[i - 1] >= (
                    num_iterations - 1
                ):  # the first frame or the last frame is completely denoised
                    new_row[i] = pre_row[i] + 1
                else:
                    new_row[i] = new_row[i - 1] - ar_step
            new_row = new_row.clamp(0, num_iterations)

            # 
            timesteps_index = num_iterations - (pre_row == 0).sum().item()
            if timesteps_index < len(timesteps_list):
                # step_template = timesteps_list[timesteps_index]
                step_matrix_cur_list = []
                for i in range(num_iterations):
                    if i < len(timesteps_list):
                        step_matrix_cur_list.append(timesteps_list[i][new_row[i]])
                        # timesteps_list tensor([1000.0000,  937.5000,  833.3333,  625.0000])
                        # timesteps_list[i] tensor(1000.)
                        # new_row tensor([1, 0, 0, 0, 0, 0, 0])
                    else:
                        step_matrix_cur_list.append(torch.tensor(999, dtype=torch.int64, device=step_template.device))
                step_matrix_cur =  torch.stack(step_matrix_cur_list, dim=0)
            else:
                step_matrix_cur_list = []
                for i in range(num_iterations):
                    if i <  timesteps_index - len(timesteps_list)  :
                        step_matrix_cur_list.append(torch.tensor(0, dtype=torch.int64, device=step_template.device))
                    elif i < timesteps_index :
                        step_matrix_cur_list.append(timesteps_list[i-(timesteps_index - len(timesteps_list))][-2])
                    else:
                        step_matrix_cur_list.append(torch.tensor(999, dtype=torch.int64, device=step_template.device))
                step_matrix_cur = torch.stack(step_matrix_cur_list, dim=0)

     
            zero_tensor = torch.tensor(0, device=device)
            nine_nine_nine = torch.tensor(999, device=device)

            current_update_mask = (new_row != pre_row) & (new_row != num_iterations)
            last_tensor_mask = (step_matrix_cur != nine_nine_nine) & (step_matrix_cur != zero_tensor)

           
            current_update_mask = current_update_mask.to(last_tensor_mask)
            update_mask.append(current_update_mask | last_tensor_mask)
            step_index.append(new_row)
            step_matrix.append(step_matrix_cur)

            pre_row = new_row

    
        def limit_single_consecutive_true_block(update_mask, max_consecutive=4):
            """
            确保每个位置有且只有一段连续True（最多max_consecutive次），之后永久为False
            参数:
                update_mask: List[torch.Tensor] - 布尔张量列表
                max_consecutive: int - 最大允许连续True次数
            返回:
                处理后的update_mask列表
            """
            if not update_mask:
                return update_mask
            
            device = update_mask[0].device
            num_positions = update_mask[0].shape[0]
            
     
            position_states = torch.zeros(num_positions, dtype=torch.int32, device=device)
            consecutive_counts = torch.zeros(num_positions, dtype=torch.int32, device=device)
            new_update_mask = []
            
            for mask in update_mask:
                current_mask = mask.clone()
                
           
                for state in [0, 1, 2]:
                    state_mask = (position_states == state)
                    if not state_mask.any():
                        continue
                        
                    if state == 0:  
                        current_active = current_mask[state_mask]
                        consecutive_counts[state_mask] += current_active.int()
                        
                        # 检查是否开始活跃段
                        new_active = consecutive_counts[state_mask] > 0
                        position_states[state_mask] = torch.where(
                            new_active, 
                            torch.ones_like(position_states[state_mask]),
                            position_states[state_mask]
                        )
                        
                    elif state == 1:  
                        current_active = current_mask[state_mask]
                        consecutive_counts[state_mask] += current_active.int()
                        
                      
                        over_limit = consecutive_counts[state_mask] > max_consecutive
                        current_mask[state_mask] &= ~over_limit
                        consecutive_counts[state_mask][over_limit] = max_consecutive
                        
                       
                        just_inactive = ~current_mask[state_mask] & (consecutive_counts[state_mask] > 0)
                        position_states[state_mask] = torch.where(
                            just_inactive,
                            torch.full_like(position_states[state_mask], 2),
                            position_states[state_mask]
                        )
                        
                    elif state == 2: 
                        current_mask[state_mask] = False
                
                new_update_mask.append(current_mask)
            
            return new_update_mask
        
        
        if constrain_mask == True:
            processed_update_mask = limit_single_consecutive_true_block(update_mask, max_consecutive=timesteps_list[0].shape[0] - 1)
            update_mask = processed_update_mask # real distillation


        # for long video we split into several sequences, base_num_frames is set to the model max length (for training)
        terminal_flag = base_num_frames_block
        if shrink_interval_with_mask:
            idx_sequence = torch.arange(num_frames_block, dtype=torch.int64)
            update_mask = update_mask[0]
            update_mask_idx = idx_sequence[update_mask]
            last_update_idx = update_mask_idx[-1].item()
            terminal_flag = last_update_idx + 1
        # for i in range(0, len(update_mask)):
        for curr_mask in update_mask:
            if terminal_flag < num_frames_block and curr_mask[terminal_flag]:
                terminal_flag += 1
            valid_interval.append((max(terminal_flag - base_num_frames_block, 0), terminal_flag))

        step_update_mask = torch.stack(update_mask, dim=0)
        step_index = torch.stack(step_index, dim=0)
        step_matrix = torch.stack(step_matrix, dim=0)

        if casual_block_size > 1:
            step_update_mask = step_update_mask.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_index = step_index.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            step_matrix = step_matrix.unsqueeze(-1).repeat(1, 1, casual_block_size).flatten(1).contiguous()
            valid_interval = [(s * casual_block_size, e * casual_block_size) for s, e in valid_interval]

        # step_num_type = 
        step_matrix = step_matrix[:-(step_num_type+1)]
        step_index = step_index[:-(step_num_type+1)]
        step_update_mask = step_update_mask[:-(step_num_type+1)] # 25s # without update mask


        return step_matrix, step_index, step_update_mask, valid_interval


    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        num_overlap_frames=None,
        full_budget=False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        num_output_frames = num_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block
                output[:, :num_overlap_frames] = initial_latent[:, :num_overlap_frames]

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        original = True
        if initial_latent is not None:
            all_num_frames = all_num_frames[int(num_overlap_frames/3):]
    
        if original:
            for block_index, current_num_frames in enumerate(all_num_frames):
                # print(block_index)
                if profile:
                    block_start.record()

                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]



                if self.args.use_diagonal_denoising :
                    if block_index == 0:
                        denoising_step_list = [int(self.denoising_step_list[0]), 933, 833, int(self.denoising_step_list[-1])]
                    elif block_index == 1:
                        denoising_step_list = [int(self.denoising_step_list[0]), 933, int(self.denoising_step_list[-1])]
                    else:
                        denoising_step_list = self.denoising_step_list
                else:
                    denoising_step_list = self.denoising_step_list





                if initial_latent is not None:
                 
                    denoising_step_list = self.denoising_step_list
    

                # 
                for index, current_timestep in enumerate(denoising_step_list):
                    print(f"current_timestep: {current_timestep}")
                    # set current timestep
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if index < len(denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input, # noise.shape
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )

                # Step 3.2: record the model's output
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                if block_index == 0 and initial_latent is None:
                    context_noise = self.args.context_noise 
                    context_timestep = torch.ones_like(timestep) * context_noise
                    # # add context noise
                    denoised_pred = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        context_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                    # with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

                if profile:
                    block_end.record()
                    torch.cuda.synchronize()
                    block_time = block_start.elapsed_time(block_end)
                    block_times.append(block_time)

                # Step 3.4: update the start and end frame indices
                current_start_frame += current_num_frames

        else:
            noisy_input_store = noise.clone()
            # step_update_mask[:,3:] = False

            latent_length = num_frames
            init_timesteps = self.denoising_step_list
            base_num_frames = num_frames
            ar_step = 1 
            predix_video_latent_length = 0
            causal_block_size = all_num_frames[0]

            step_matrix, _, step_update_mask, valid_interval = self.generate_timestep_matrix_DD(
                latent_length, init_timesteps, base_num_frames, ar_step, predix_video_latent_length, causal_block_size, constrain_mask=True,
            )
            
            for i, timestep_i in enumerate((step_matrix)):
                update_mask_i = step_update_mask[i]
                # timestep = timestep_i
                update_mask_index = (torch.nonzero(update_mask_i).flatten()[::causal_block_size] / causal_block_size).tolist()
                # print(update_mask_index)

                for temp_idx , block_index in enumerate(update_mask_index):
                    if update_mask_index[0] == 0 or (update_mask_index[0] != 0 and temp_idx == (len(self.denoising_step_list) - 1) ):

                        block_index = int(block_index)
                        timestep = timestep_i[block_index * causal_block_size: (block_index + 1) * causal_block_size].unsqueeze(0).to(noisy_input_store)
                        noisy_input = noisy_input_store[:, block_index *
                                        causal_block_size:(block_index + 1) * causal_block_size]  

                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        
                        if i < (step_matrix.shape[0] - 1):
                            next_timestep = step_matrix[i+1][(block_index ) * causal_block_size: (block_index + 1) * causal_block_size].to(noisy_input_store)
                        else:
                            next_timestep = torch.zeros_like(next_timestep).to(noisy_input_store)

                        if len(update_mask_index) == len(self.denoising_step_list):
                            next_timestep = torch.zeros_like(next_timestep).to(noisy_input_store)

                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * # next
                            torch.ones([batch_size], device="cuda",
                                        dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                        print("block_index:", block_index)
                        print("timestep:", timestep)
                        print("next_timestep:", next_timestep)

                        noisy_input_store[:, block_index * causal_block_size : (block_index + 1) * causal_block_size] = noisy_input.clone()

                        if len(update_mask_index) == 3:
                         
                            self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        
            output = noisy_input_store 





        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
