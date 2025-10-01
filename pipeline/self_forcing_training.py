from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist
import random


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        # self.same_step_across_blocks = False
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()


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
        import math
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
        # timesteps_list = self.denoising_step_list

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

            # if 
            # 预定义常量张量（避免重复创建）
            zero_tensor = torch.tensor(0, device=device)
            nine_nine_nine = torch.tensor(999, device=device)

            # 合并所有条件计算（减少内存访问）
            current_update_mask = (new_row != pre_row) & (new_row != num_iterations)
            last_tensor_mask = (step_matrix_cur != nine_nine_nine) & (step_matrix_cur != zero_tensor)

            # 一次性完成所有操作（避免列表追加和索引访问）
            current_update_mask = current_update_mask.to(last_tensor_mask)
            update_mask.append(current_update_mask | last_tensor_mask)
            step_index.append(new_row)
            step_matrix.append(step_matrix_cur)

            pre_row = new_row

        # 对于update_mask 列表中每个元素，同一个位置最多只能连续为True 4次

        # post_process_step_matrix
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
            
            # 状态跟踪器：0=未开始, 1=活跃中, 2=已结束
            position_states = torch.zeros(num_positions, dtype=torch.int32, device=device)
            consecutive_counts = torch.zeros(num_positions, dtype=torch.int32, device=device)
            new_update_mask = []
            
            for mask in update_mask:
                current_mask = mask.clone()
                
                # 处理三种状态的位置
                for state in [0, 1, 2]:
                    state_mask = (position_states == state)
                    if not state_mask.any():
                        continue
                        
                    if state == 0:  # 未开始的位置
                        current_active = current_mask[state_mask]
                        consecutive_counts[state_mask] += current_active.int()
                        
                        # 检查是否开始活跃段
                        new_active = consecutive_counts[state_mask] > 0
                        position_states[state_mask] = torch.where(
                            new_active, 
                            torch.ones_like(position_states[state_mask]),
                            position_states[state_mask]
                        )
                        
                    elif state == 1:  # 活跃中的位置
                        current_active = current_mask[state_mask]
                        consecutive_counts[state_mask] += current_active.int()
                        
                        # 检查是否达到限制
                        over_limit = consecutive_counts[state_mask] > max_consecutive
                        current_mask[state_mask] &= ~over_limit
                        consecutive_counts[state_mask][over_limit] = max_consecutive
                        
                        # 检查是否结束活跃段
                        just_inactive = ~current_mask[state_mask] & (consecutive_counts[state_mask] > 0)
                        position_states[state_mask] = torch.where(
                            just_inactive,
                            torch.full_like(position_states[state_mask], 2),
                            position_states[state_mask]
                        )
                        
                    elif state == 2:  # 已结束的位置
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



    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            current_step=None,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
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
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        # if current_step is None:
        #     current_step = 1

        # random_denoising = current_step % 2
        # if random_denoising == 1:
        #     num_denoising_steps = len(self.denoising_step_list)
        # else:
        #     num_denoising_steps = 2

        # num_denoising_steps = len(self.denoising_step_list)

        num_denoising_steps = len(self.denoising_step_list)
        
        # num_denoising_steps = 2
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        random_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)

        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        original = True
        
        if original:
            for block_index, current_num_frames in enumerate(all_num_frames):
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # if block_index == 0:
                #     denoising_step_list = self.denoising_step_list[:(len(self.denoising_step_list) - block_index)]
                # else:
                #     denoising_step_list = self.denoising_step_list[::3]
                if block_index == 0:
                    denoising_step_list = [int(self.denoising_step_list[0]), 933, 833, int(self.denoising_step_list[-1])]
                elif block_index == 1:
                    denoising_step_list = [int(self.denoising_step_list[0]), 933, int(self.denoising_step_list[-1])]
                else:
                    denoising_step_list = self.denoising_step_list
       

                # Step 3.1: Spatial denoising loop
                # if block_index == 


                for index, current_timestep in enumerate(denoising_step_list):
                    if self.same_step_across_blocks:
                        exit_flag = (index == exit_flags[0])
                    else:
                        exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)

                    if block_index == 0:
                        exit_flag = (index == len(denoising_step_list)-1)
                    
                    if block_index == 1:
                        exit_flag = (index == len(denoising_step_list) - 2 )


                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
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
                        # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                        if current_start_frame < start_gradient_frame_index:
                            with torch.no_grad():
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length
                                )
                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        break

                # Step 3.2: record the model's output
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                # Step 3.3: rerun with timestep zero to update the cache


                # if random_denoising == 0:
                #     if block_index != 0:
                #         self.context_noise = int(self.denoising_step_list[-1])
                #     else:
                #         self.context_noise = 0
                # else:
                #     # 有take random noise 和 clean frame 作为condition的能力
                #     temp_denoising_step_list = self.denoising_step_list
                #     temp_denoising_step_list[0] = 0
                #     self.context_noise = int(temp_denoising_step_list[exit_flags[block_index]])
                # self.context_noise = 0

                # 方案1: 类causvid
                # if block_index == 0:
                #     self.context_noise = 0
                # else:
                #     if exit_flags[block_index] == 0:
                #         self.context_noise = 0
                #     else:      
                #         self.context_noise = int(self.denoising_step_list[-1])


                # 方案2: diagonal denoising and causvid
                # /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/20250623_000310 250
                # 
                if block_index == 0:
                    self.context_noise = 0
                else:
                    if exit_flags[0] == 0:
                        # case 1
                        if random_flags[0] == 0:
                            if block_index == 0:
                                self.context_noise = int(self.denoising_step_list[1])
                            else:
                                if exit_flags[block_index] == 0:
                                    self.context_noise = 0
                                else:      
                                    self.context_noise = int(self.denoising_step_list[-1])
                        else:
                            if block_index == 0:
                                self.context_noise = int(self.denoising_step_list[2])
                            elif block_index == 1:
                                self.context_noise = int(self.denoising_step_list[1])
                            else:
                                if exit_flags[block_index] == 0:
                                    self.context_noise = 0
                                else:      
                                    self.context_noise = int(self.denoising_step_list[-1])
                    else:
                        # 后续部分，多种condition 在一个模型中大一统？
                        if exit_flags[block_index] == 0:
                            self.context_noise = 0
                        else:      
                            self.context_noise = int(self.denoising_step_list[-1])

                # 方案3： most simple
                # 250 /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/20250623_181405 (头逐渐变大的问题)
                # 100 /mnt/bn/jinxiuliu-hl/Long_Video_Gen/causal_distill/Self-Forcing/logs/self_forcing_dmd/20250624_172848
                if block_index == 0:
                    self.context_noise = 0
                else:      
                    self.context_noise = int(self.denoising_step_list[-1])


                # 方案2: 对角去噪友好

                # if block_index == 0:
                #     self.context_noise = 0
                # else:
                #     exit_flags = [1,0,0,0,0,0,0]
                #     if exit_flags[block_index] == 0:
                #         self.context_noise = 0
                #     else:      
                #         self.context_noise = int(self.denoising_step_list[-1])

                #     exit_flags = [2,1,0,0,0,0,0]

                #     exit_flags = [3,2,1,0,0,0,0]


                # 方案3: 


                context_timestep = torch.ones_like(timestep) * self.context_noise
                # add context noise
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    context_timestep * torch.ones(
                        [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

                # Step 3.4: update the start and end frame indices
                current_start_frame += current_num_frames



            # for block_index, current_num_frames in enumerate(all_num_frames):
            #     noisy_input = noise[
            #         :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            #     # if block_index < len(self.denoising_step_list) - 1:
            #     #     denoising_step_list = self.denoising_step_list[:(len(self.denoising_step_list) - block_index)]
            #     # else:
            #     #     # denoising_step_list = self.denoising_step_list[:1]
            #     #     denoising_step_list = self.denoising_step_list[::3]

            #     # if block_index == 0:
            #     #     denoising_step_list = self.denoising_step_list[:(len(self.denoising_step_list) - block_index)]
            #     # else:
            #     #     denoising_step_list = self.denoising_step_list[:1]
            #     #     denoising_step_list = denoising_step_list[::3]

                # if block_index == 0:
                #     denoising_step_list = self.denoising_step_list[:(len(self.denoising_step_list) - block_index)]
                # else:
                #     denoising_step_list = self.denoising_step_list[::3]

            #     # # 传入timestep
            #     # if random_denoising == 1:  # 50%概率执行原始逻辑
            #     #     if block_index < len(self.denoising_step_list) - 1:
            #     #         denoising_step_list = self.denoising_step_list[:(len(self.denoising_step_list) - block_index)]
            #     #     else:
            #     #         denoising_step_list = self.denoising_step_list[::3]
            #     # else:  
            #     #     if block_index == 0:
            #     #         denoising_step_list = self.denoising_step_list[:(len(self.denoising_step_list) - block_index)]
            #     #     else:
            #     #         denoising_step_list = self.denoising_step_list[::3]
                
            #     # if exit_flag = (block == exit_flags[0])

            #     # Step 3.1: Spatial denoising loop
            #     # for index, current_timestep in enumerate(self.denoising_step_list):

            #     for index, current_timestep in enumerate(denoising_step_list):
            #         if self.same_step_across_blocks:
            #             exit_flag = (index == exit_flags[0])
            #         else:
            #             exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
            #         timestep = torch.ones(
            #             [batch_size, current_num_frames],
            #             device=noise.device,
            #             dtype=torch.int64) * current_timestep

            #         if not exit_flag:
            #             with torch.no_grad():
            #                 _, denoised_pred = self.generator(
            #                     noisy_image_or_video=noisy_input,
            #                     conditional_dict=conditional_dict,
            #                     timestep=timestep,
            #                     kv_cache=self.kv_cache1,
            #                     crossattn_cache=self.crossattn_cache,
            #                     current_start=current_start_frame * self.frame_seq_length
            #                 )
            #                 next_timestep = self.denoising_step_list[index + 1]
            #                 noisy_input = self.scheduler.add_noise(
            #                     denoised_pred.flatten(0, 1),
            #                     torch.randn_like(denoised_pred.flatten(0, 1)),
            #                     next_timestep * torch.ones(
            #                         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            #                 ).unflatten(0, denoised_pred.shape[:2])
            #         else:
            #             # for getting real output
            #             # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
            #             if current_start_frame < start_gradient_frame_index:
            #                 with torch.no_grad():
            #                     _, denoised_pred = self.generator(
            #                         noisy_image_or_video=noisy_input,
            #                         conditional_dict=conditional_dict,
            #                         timestep=timestep,
            #                         kv_cache=self.kv_cache1,
            #                         crossattn_cache=self.crossattn_cache,
            #                         current_start=current_start_frame * self.frame_seq_length
            #                     )
            #             else:
            #                 _, denoised_pred = self.generator(
            #                     noisy_image_or_video=noisy_input,
            #                     conditional_dict=conditional_dict,
            #                     timestep=timestep,
            #                     kv_cache=self.kv_cache1,
            #                     crossattn_cache=self.crossattn_cache,
            #                     current_start=current_start_frame * self.frame_seq_length
            #                 )
            #             break

            #     # 选用不同的KV Cache，多级KV Cache 存储，调用不同的KV Cache组合

            #     # Step 3.2: record the model's output
            #     output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            #     # Step 3.3: rerun with timestep zero to update the cache
            #     # 根据block index 有关，随机取值，直接随机取值 -2，-1
            #     # 直接改为random noise level ？

            #     # if random_denoising != 1:
            #     #     if block_index % 2 == 0:
            #     #         self.context_noise = int(self.denoising_step_list[-1])
            #     #     else:
            #     #         self.context_noise = 0
            #     # else:
            #     #     # if case_1:
            #     #     if block_index % 2 == 1:
            #     #         self.context_noise = int(self.denoising_step_list[-1])
            #     #     else:
            #     #         self.context_noise = 0
                
                # if random_denoising == 0:
                #     if block_index != 0:
                #         self.context_noise = int(self.denoising_step_list[-1])
                #     else:
                #         self.context_noise = 0
                # else:
                #     # 有take random noise 和 clean frame 作为condition的能力
                #     temp_denoising_step_list = self.denoising_step_list
                #     temp_denoising_step_list[0] = 0
                #     self.context_noise = int(temp_denoising_step_list[exit_flags[block_index]])


            #     self.context_noise = 0

            #     context_timestep = torch.ones_like(timestep) * self.context_noise
            #     # add context noise
            #     denoised_pred = self.scheduler.add_noise(
            #         denoised_pred.flatten(0, 1),
            #         torch.randn_like(denoised_pred.flatten(0, 1)),
            #         context_timestep * torch.ones(
            #             [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            #     ).unflatten(0, denoised_pred.shape[:2])
            
            #     with torch.no_grad():
            #         self.generator(
            #             noisy_image_or_video=denoised_pred,
            #             conditional_dict=conditional_dict,
            #             timestep=context_timestep,
            #             kv_cache=self.kv_cache1,
            #             crossattn_cache=self.crossattn_cache,
            #             current_start=current_start_frame * self.frame_seq_length
            #         )

            #     # Step 3.4: update the start and end frame indices
            #     current_start_frame += current_num_frames
        else:

            noisy_input_store = noise
            # step_update_mask[:,3:] = False

            latent_length = num_frames
            init_timesteps = self.denoising_step_list
            base_num_frames = num_frames
            ar_step = 1 # 3, 不是严格对角去噪
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
                exit_flag = True
                for temp_idx , block_index in enumerate(update_mask_index):
                    if update_mask_index[0] == 0 or (update_mask_index[0] != 0 and temp_idx == (len(self.denoising_step_list) - 1) ):

                        block_index = int(block_index)
                        timestep = timestep_i[block_index * causal_block_size: (block_index + 1) * causal_block_size].unsqueeze(0).to(noisy_input_store)
                        noisy_input = noisy_input_store[:, block_index *
                                        causal_block_size:(block_index + 1) * causal_block_size]  


                        if not exit_flag:
                            with torch.no_grad():
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
                        else:
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



                        noisy_input_store[:, block_index * causal_block_size : (block_index + 1) * causal_block_size] = noisy_input

                        if len(update_mask_index) == 3:
                            # 不更新kv cache
                            with torch.no_grad():
                                self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length
                                )
                        
            output = noisy_input_store 




        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
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
