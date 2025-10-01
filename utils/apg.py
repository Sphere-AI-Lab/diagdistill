"""
Adaptive Projected Guidance (APG) Implementation

This module implements Adaptive Projected Guidance, which addresses oversaturation
and artifacts in Classifier-Free Guidance (CFG) at high guidance scales.

Based on the paper: "Adaptive Projected Guidance for Diffusion Models"
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class APGuidance:
    """
    Adaptive Projected Guidance implementation.

    APG improves CFG by:
    1. Orthogonal projection: Separating parallel and orthogonal components
    2. Rescaling: Constraining updates within a sphere radius
    3. Reverse momentum: Using negative momentum to avoid previous directions
    """

    def __init__(
        self,
        eta: float = 0.1,
        rescale_radius: Optional[float] = None,
        momentum_beta: float = -0.1,
        device: str = "cuda"
    ):
        """
        Initialize APG parameters.

        Args:
            eta: Strength of parallel component (≤ 1.0). Lower values reduce saturation.
            rescale_radius: Radius for constraining update magnitude. None disables rescaling.
            momentum_beta: Negative momentum strength (< 0). More negative = stronger effect.
            device: Device for computations.
        """
        self.eta = eta
        self.rescale_radius = rescale_radius
        self.momentum_beta = momentum_beta
        self.device = device

        # Momentum state - stores previous update direction
        self.momentum_state = None

    def reset_momentum(self):
        """Reset momentum state for new generation sequence."""
        self.momentum_state = None

    def orthogonal_projection(
        self,
        update_direction: torch.Tensor,
        conditional_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose update direction into parallel and orthogonal components.

        Args:
            update_direction: CFG update direction Δ_D_t = D_cond - D_uncond
            conditional_pred: Conditional prediction D_θ(z_t, t, y)

        Returns:
            Tuple of (orthogonal_component, parallel_component)
        """
        # Flatten for easier computation
        orig_shape = update_direction.shape
        update_flat = update_direction.flatten()
        cond_flat = conditional_pred.flatten()

        # Compute parallel component: projection of update onto conditional prediction
        dot_product = torch.dot(update_flat, cond_flat)
        norm_squared = torch.dot(cond_flat, cond_flat)

        # Avoid division by zero
        if norm_squared > 1e-8:
            parallel_component = (dot_product / norm_squared) * cond_flat
        else:
            parallel_component = torch.zeros_like(update_flat)

        # Orthogonal component is the remainder
        orthogonal_component = update_flat - parallel_component

        # Reshape back to original shape
        parallel_component = parallel_component.reshape(orig_shape)
        orthogonal_component = orthogonal_component.reshape(orig_shape)

        return orthogonal_component, parallel_component

    def apply_rescaling(self, update_direction: torch.Tensor) -> torch.Tensor:
        """
        Apply rescaling to constrain update within sphere radius.

        Args:
            update_direction: Update direction to rescale

        Returns:
            Rescaled update direction
        """
        if self.rescale_radius is None:
            return update_direction

        # Compute current magnitude
        update_norm = torch.norm(update_direction)

        # Apply rescaling if magnitude exceeds radius
        if update_norm > self.rescale_radius:
            scale_factor = self.rescale_radius / (update_norm + 1e-8)
            return update_direction * scale_factor

        return update_direction

    def apply_momentum(self, update_direction: torch.Tensor) -> torch.Tensor:
        """
        Apply reverse momentum to the update direction.

        Args:
            update_direction: Current update direction

        Returns:
            Update direction with momentum applied
        """
        if self.momentum_state is None:
            # First step - no momentum to apply
            self.momentum_state = update_direction.clone()
            return update_direction

        # Apply negative momentum: push away from previous direction
        momentum_term = self.momentum_beta * self.momentum_state
        updated_direction = update_direction + momentum_term

        # Update momentum state for next iteration
        self.momentum_state = update_direction.clone()

        return updated_direction

    def __call__(
        self,
        conditional_pred: torch.Tensor,
        unconditional_pred: torch.Tensor,
        guidance_scale: float
    ) -> torch.Tensor:
        """
        Apply APG to conditional and unconditional predictions.

        Args:
            conditional_pred: D_θ(z_t, t, y) - conditional model output
            unconditional_pred: D_θ(z_t, t, ∅) - unconditional model output
            guidance_scale: Guidance scale w

        Returns:
            APG-guided prediction
        """
        # Step 1: Compute CFG update direction
        delta_D = conditional_pred - unconditional_pred

        # Step 2: Orthogonal projection
        delta_D_perp, delta_D_parallel = self.orthogonal_projection(delta_D, conditional_pred)

        # Step 3: Modify update direction with reduced parallel component
        modified_delta_D = delta_D_perp + self.eta * delta_D_parallel

        # Step 4: Apply rescaling
        if self.rescale_radius is not None:
            modified_delta_D = self.apply_rescaling(modified_delta_D)

        # Step 5: Apply reverse momentum
        modified_delta_D = self.apply_momentum(modified_delta_D)

        # Step 6: Compute final APG prediction
        # APG formula: D_APG = D_cond + (w-1) * modified_delta_D
        apg_pred = conditional_pred + (guidance_scale - 1) * modified_delta_D

        return apg_pred


def apply_apg_guidance(
    conditional_pred: torch.Tensor,
    unconditional_pred: torch.Tensor,
    guidance_scale: float,
    eta: float = 0.1,
    rescale_radius: Optional[float] = None,
    momentum_beta: float = -0.1,
    apg_instance: Optional[APGuidance] = None
) -> torch.Tensor:
    """
    Convenience function to apply APG guidance.

    Args:
        conditional_pred: Conditional model prediction
        unconditional_pred: Unconditional model prediction
        guidance_scale: CFG guidance scale
        eta: Parallel component strength
        rescale_radius: Rescaling radius (None to disable)
        momentum_beta: Reverse momentum strength
        apg_instance: Optional pre-initialized APG instance

    Returns:
        APG-guided prediction
    """
    if apg_instance is None:
        apg_instance = APGuidance(
            eta=eta,
            rescale_radius=rescale_radius,
            momentum_beta=momentum_beta,
            device=conditional_pred.device
        )

    return apg_instance(conditional_pred, unconditional_pred, guidance_scale)