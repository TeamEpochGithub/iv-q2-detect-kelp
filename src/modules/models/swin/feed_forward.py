"""Feedforward network."""
import torch
from torch import nn


class FeedForward(nn.Module):
    """Feedforward network.

    :param dim: input dimension
    :param hidden_dim: hidden dimension
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Initialize the FeedForward.

        :param dim: input dimension
        :param hidden_dim: hidden dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        return self.net(x)
