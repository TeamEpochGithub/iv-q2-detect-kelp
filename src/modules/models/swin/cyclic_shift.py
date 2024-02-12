"""Cyclic shift layer."""
import torch
from torch import nn


class CyclicShift(nn.Module):
    """Cyclic shift layer.

    :param displacement: displacement to apply to the input tensor
    """

    def __init__(self, displacement: int) -> None:
        """Initialize the CyclicShift.

        :param displacement: displacement to apply to the input tensor
        """
        super().__init__()
        self.displacement = displacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))
