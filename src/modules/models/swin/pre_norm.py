"""Pre-normalization layer."""
import torch
from torch import nn


class PreNorm(nn.Module):
    """Pre-normalization layer.

    :param dim: input dimension
    :param fn: function to apply to the input tensor
    """

    def __init__(self, dim: int, fn: nn.Module) -> None:
        """Initialize the PreNorm.

        :param dim: input dimension
        :param fn: function to apply to the input tensor
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: float) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        return self.fn(self.norm(x), **kwargs)
