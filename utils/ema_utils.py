"""
EMA (Exponential Moving Average) utilities for MLP components.
This module provides comprehensive EMA support for neural network parameters,
with special focus on MLP (Multi-Layer Perceptron) components.

Reference implementation from MCM codebase with enhancements for "think harder" approach.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from collections import defaultdict


@torch.no_grad()
def update_ema(target_params, source_params, decay_rate: float = 0.99):
    """
    Update target parameters using exponential moving average.
    
    Args:
        target_params: Target parameter sequence (EMA parameters)
        source_params: Source parameter sequence (current model parameters) 
        decay_rate: EMA decay rate (closer to 1 means slower updates)
    """
    for target_param, source_param in zip(target_params, source_params):
        target_param.detach().mul_(decay_rate).add_(source_param, alpha=1 - decay_rate)


class EMAWrapper(nn.Module):
    """
    EMA wrapper for individual neural network modules.
    Maintains shadow copies of parameters and updates them with EMA.
    """
    
    def __init__(self, module: nn.Module, decay: float = 0.999, update_after: int = 100, update_every: int = 10):
        """
        Args:
            module: The neural network module to apply EMA to
            decay: EMA decay rate 
            update_after: Number of steps to wait before starting EMA updates
            update_every: Update EMA parameters every N steps
        """
        super().__init__()
        self.module = module
        self.decay = decay
        self.update_after = update_after
        self.update_every = update_every
        self.step_count = 0
        
        # Create shadow parameters
        self.shadow_params = {}
        self._create_shadow_params()
        
    def _create_shadow_params(self):
        """Create shadow copies of all module parameters."""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.detach().clone()
                
    @torch.no_grad()
    def update_parameters(self):
        """Update EMA parameters if conditions are met."""
        self.step_count += 1
        
        if self.step_count <= self.update_after:
            # Copy current parameters to shadow during warmup
            for name, param in self.module.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.shadow_params[name].copy_(param)
            return
            
        if self.step_count % self.update_every != 0:
            return
            
        # Perform EMA update
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                shadow_param = self.shadow_params[name]
                shadow_param.mul_(self.decay).add_(param, alpha=1 - self.decay)
    
    @torch.no_grad()
    def copy_params_to_ema(self):
        """Copy current parameters to EMA parameters."""
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].copy_(param)
    
    @torch.no_grad()
    def copy_ema_to_params(self):
        """Copy EMA parameters back to model parameters."""
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                param.copy_(self.shadow_params[name])
    
    def forward(self, *args, use_ema: bool = False, **kwargs):
        """
        Forward pass with option to use EMA parameters.
        
        Args:
            use_ema: If True, temporarily use EMA parameters for forward pass
        """
        if use_ema:
            # Temporarily swap to EMA parameters
            original_params = {}
            for name, param in self.module.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    original_params[name] = param.detach().clone()
                    param.copy_(self.shadow_params[name])
            
            try:
                output = self.module(*args, **kwargs)
            finally:
                # Restore original parameters
                for name, param in self.module.named_parameters():
                    if name in original_params:
                        param.copy_(original_params[name])
            
            return output
        else:
            return self.module(*args, **kwargs)


class EMAMLPWrapper(EMAWrapper):
    """
    Specialized EMA wrapper for MLP modules with enhanced functionality.
    """
    
    def __init__(self, mlp_module: nn.Module, decay: float = 0.999, 
                 update_after: int = 100, update_every: int = 10,
                 track_norms: bool = True):
        """
        Args:
            mlp_module: MLP module to wrap
            decay: EMA decay rate
            update_after: Steps before starting EMA
            update_every: Update frequency  
            track_norms: Whether to track parameter norms for monitoring
        """
        super().__init__(mlp_module, decay, update_after, update_every)
        self.track_norms = track_norms
        self.param_norms = defaultdict(list) if track_norms else None
        
    @torch.no_grad()
    def update_parameters(self):
        """Update EMA parameters and optionally track norms."""
        super().update_parameters()
        
        if self.track_norms and self.step_count > self.update_after:
            # Track parameter norms for monitoring
            for name, param in self.module.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    param_norm = param.norm().item()
                    ema_norm = self.shadow_params[name].norm().item()
                    self.param_norms[name].append({
                        'step': self.step_count,
                        'param_norm': param_norm,
                        'ema_norm': ema_norm,
                        'norm_ratio': ema_norm / (param_norm + 1e-8)
                    })
                    
                    # Keep only recent history
                    if len(self.param_norms[name]) > 1000:
                        self.param_norms[name] = self.param_norms[name][-1000:]
    
    def get_norm_stats(self) -> Dict[str, Any]:
        """Get parameter norm statistics."""
        if not self.track_norms or not self.param_norms:
            return {}
            
        stats = {}
        for name, history in self.param_norms.items():
            if history:
                recent = history[-10:] if len(history) >= 10 else history
                param_norms = [h['param_norm'] for h in recent]
                ema_norms = [h['ema_norm'] for h in recent]
                norm_ratios = [h['norm_ratio'] for h in recent]
                
                stats[name] = {
                    'param_norm_mean': sum(param_norms) / len(param_norms),
                    'ema_norm_mean': sum(ema_norms) / len(ema_norms), 
                    'norm_ratio_mean': sum(norm_ratios) / len(norm_ratios),
                    'updates': len(history)
                }
        
        return stats


class EMAManager:
    """
    Manager for multiple EMA-wrapped modules with centralized control.
    """
    
    def __init__(self, decay: float = 0.999, update_after: int = 100, update_every: int = 10):
        """
        Args:
            decay: Default EMA decay rate
            update_after: Default steps before starting EMA
            update_every: Default update frequency
        """
        self.decay = decay
        self.update_after = update_after  
        self.update_every = update_every
        self.wrapped_modules: Dict[str, EMAWrapper] = {}
        
    def wrap_module(self, name: str, module: nn.Module, 
                   decay: Optional[float] = None,
                   update_after: Optional[int] = None,
                   update_every: Optional[int] = None,
                   is_mlp: bool = True) -> EMAWrapper:
        """
        Wrap a module with EMA.
        
        Args:
            name: Unique identifier for the module
            module: Module to wrap
            decay: EMA decay rate (uses default if None)
            update_after: Steps before EMA starts (uses default if None)
            update_every: Update frequency (uses default if None)  
            is_mlp: Use specialized MLP wrapper if True
            
        Returns:
            EMA wrapped module
        """
        decay = decay if decay is not None else self.decay
        update_after = update_after if update_after is not None else self.update_after
        update_every = update_every if update_every is not None else self.update_every
        
        if is_mlp:
            wrapped = EMAMLPWrapper(module, decay, update_after, update_every)
        else:
            wrapped = EMAWrapper(module, decay, update_after, update_every)
            
        self.wrapped_modules[name] = wrapped
        return wrapped
    
    def update_all(self):
        """Update EMA parameters for all wrapped modules."""
        for wrapper in self.wrapped_modules.values():
            wrapper.update_parameters()
    
    def copy_params_to_ema_all(self):
        """Copy current parameters to EMA for all modules.""" 
        for wrapper in self.wrapped_modules.values():
            wrapper.copy_params_to_ema()
            
    def copy_ema_to_params_all(self):
        """Copy EMA parameters to current parameters for all modules."""
        for wrapper in self.wrapped_modules.values():
            wrapper.copy_ema_to_params()
    
    def get_all_norm_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get norm statistics for all MLP modules."""
        all_stats = {}
        for name, wrapper in self.wrapped_modules.items():
            if isinstance(wrapper, EMAMLPWrapper):
                stats = wrapper.get_norm_stats()
                if stats:
                    all_stats[name] = stats
        return all_stats
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for all EMA parameters.""" 
        state = {}
        for name, wrapper in self.wrapped_modules.items():
            state[name] = {
                'shadow_params': wrapper.shadow_params,
                'step_count': wrapper.step_count,
                'decay': wrapper.decay,
                'update_after': wrapper.update_after,
                'update_every': wrapper.update_every
            }
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dictionary for EMA parameters."""
        for name, wrapper_state in state.items():
            if name in self.wrapped_modules:
                wrapper = self.wrapped_modules[name]
                wrapper.shadow_params = wrapper_state['shadow_params']
                wrapper.step_count = wrapper_state['step_count']
                wrapper.decay = wrapper_state['decay']
                wrapper.update_after = wrapper_state['update_after']
                wrapper.update_every = wrapper_state['update_every']


def create_ema_mlp_from_sequential(sequential_module: nn.Sequential, 
                                  decay: float = 0.999,
                                  update_after: int = 100, 
                                  update_every: int = 10) -> EMAMLPWrapper:
    """
    Convenience function to create EMA wrapper for Sequential MLP modules.
    
    Args:
        sequential_module: nn.Sequential module (typically MLP)
        decay: EMA decay rate
        update_after: Steps before starting EMA
        update_every: Update frequency
        
    Returns:
        EMA wrapped module
    """
    return EMAMLPWrapper(sequential_module, decay, update_after, update_every, track_norms=True)


def apply_ema_to_model_mlps(model: nn.Module, 
                           ema_manager: EMAManager,
                           mlp_names: List[str] = None,
                           decay: float = 0.999) -> Dict[str, EMAWrapper]:
    """
    Automatically detect and wrap MLP components in a model.
    
    Args:
        model: PyTorch model
        ema_manager: EMA manager instance  
        mlp_names: Specific module names to wrap (auto-detect if None)
        decay: EMA decay rate
        
    Returns:
        Dictionary of wrapped modules
    """
    wrapped = {}
    
    if mlp_names is None:
        # Auto-detect MLP modules
        mlp_names = []
        for name, module in model.named_modules():
            # Check for common MLP patterns
            if (isinstance(module, nn.Sequential) and 
                any(isinstance(layer, nn.Linear) for layer in module) and
                len([l for l in module if isinstance(l, nn.Linear)]) >= 2):
                mlp_names.append(name)
            elif name.endswith(('ffn', 'mlp', 'proj', 'embedding')) and hasattr(module, 'parameters'):
                mlp_names.append(name)
    
    # Wrap identified modules
    for name in mlp_names:
        try:
            module = model.get_submodule(name)
            wrapped_module = ema_manager.wrap_module(
                name, module, decay=decay, is_mlp=True
            )
            wrapped[name] = wrapped_module
            
            # Replace module in model with wrapped version
            parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, wrapped_module)
            else:
                setattr(model, child_name, wrapped_module)
                
        except Exception as e:
            print(f"Warning: Failed to wrap module {name}: {e}")
    
    return wrapped