import torch
import torch.nn as nn
from einops import rearrange


class SpatialHead(nn.Module):
    """Frame-wise latent projector without temporal layers.

    Input/Output shape: [B, C, T, H, W].
    """

    def __init__(
        self,
        num_channels: int,
        num_layers: int = 2,
        kernel_size: int = 1,
        hidden_dim: int = 64,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.in_act = nn.SiLU()
        padding = (kernel_size - 1) // 2
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        num_channels if i == 0 else hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=padding,
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

        # Start from near-identity behavior, as done in MCM.
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> frame-wise 2D projection.
        b, c, t, h, w = x.shape
        x_in = x
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.in_act(x)
        for layer in self.layers:
            x = layer(x)
        x = self.conv_out(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return x + x_in

