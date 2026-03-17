from typing import Iterable, Optional

import torch


def resolve_denoising_steps(
    raw_steps: Iterable[int],
    scheduler_timesteps: torch.Tensor,
    warp_denoising_step: bool,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert configured denoising steps to runtime scheduler domain."""
    steps = torch.tensor(list(raw_steps), dtype=torch.long)
    if warp_denoising_step:
        timesteps = torch.cat(
            (scheduler_timesteps.detach().cpu(), torch.tensor([0], dtype=torch.float32))
        )
        steps = timesteps[1000 - steps]
    if device is not None:
        steps = steps.to(device)
    return steps


def build_block_denoising_steps(
    base_steps: torch.Tensor,
    block_index: int,
    use_diagonal_denoising: bool,
    warmup_mid_steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build per-block denoising schedule.
    - block 0: 4-step (first + 2 mids + last)
    - block 1: 3-step (first + 1 mid + last)
    - block >=2: 2-step (first + last)
    """
    if (not use_diagonal_denoising) or base_steps.numel() <= 1:
        return base_steps

    first = base_steps[0:1]
    last = base_steps[-1:]
    mids = base_steps[1:-1]

    if warmup_mid_steps is not None and warmup_mid_steps.numel() > 0:
        mids = warmup_mid_steps.to(device=base_steps.device, dtype=base_steps.dtype)

    if block_index <= 0:
        if mids.numel() >= 2:
            selected_mid = mids[:2]
        elif mids.numel() == 1:
            selected_mid = mids
        else:
            selected_mid = torch.empty(0, dtype=base_steps.dtype, device=base_steps.device)
        return torch.cat([first, selected_mid, last], dim=0)

    if block_index == 1:
        if mids.numel() >= 1:
            return torch.cat([first, mids[:1], last], dim=0)
        return torch.cat([first, last], dim=0)

    return torch.cat([first, last], dim=0)
