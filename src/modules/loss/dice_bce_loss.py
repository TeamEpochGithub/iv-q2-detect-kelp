"""Implementation of Dice Loss + BCE for image segmentation."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiceBCELoss(nn.Module):
    """Dice BCE loss."""

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :param smooth: smoothing factor
        :return: loss
        """
        # flatten label and prediction tensors
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction="mean")
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = 1 - ((2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))
        return dice + bce
