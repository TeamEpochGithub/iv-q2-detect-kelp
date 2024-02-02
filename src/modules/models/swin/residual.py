"""Residual connection."""

import torch
from torch import nn


class Residual(nn.Module):
    """Residual connection.

    :param sublayer: sublayer to apply the residual connection to
    """

    def __init__(self, sublayer: nn.Module) -> None:
        """Initialize the Residual.

        :param sublayer: sublayer to apply the residual connection to
        """
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, **kwargs: float) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        return self.sublayer(x, **kwargs) + x
