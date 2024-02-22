"""Implementation of Dice Loss for image segmentation."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiceLoss(nn.Module):
    """Dice loss, also known as soft Sorenson-Dice loss or Tversky loss."""

    multiply_kelp: bool = False

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
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        # Number of 1s in target
        num_ones = targets.sum() if self.multiply_kelp else 1.0
        return (1 - dice) * num_ones
