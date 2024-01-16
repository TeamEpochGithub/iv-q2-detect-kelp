"""Implementation of Shrinkage Loss."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ShrinkageLoss(nn.Module):
    """Shrinkage Loss.

    :param reduction: Reduction mode for the loss. Options are 'mean' and 'sum'.
    """

    reduction: str = "mean"

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, alpha: int = 2, c: float = 0.2) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :param alpha: alpha parameter
        :param c: c parameter
        :return: loss
        """
        l1_loss = torch.abs(inputs - targets)
        shrinkage_loss = (l1_loss**2) * torch.exp(targets) / (1 + torch.exp(alpha * (c - l1_loss)))
        if self.reduction == "mean":
            return shrinkage_loss.mean()
        return shrinkage_loss
