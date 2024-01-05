"""Implementation of Dice Loss for image segmentation."""

import torch
from torch import nn


class DiceLoss(nn.Module):
    """Dice loss, also known as soft Sorenson-Dice loss or Tversky loss.

    :param size_average: If True, loss is averaged over the batch. If False, loss is summed over the batch.
    """

    def __init__(self) -> None:
        """Initialize the DiceLoss."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> float:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :param smooth: smoothing factor
        :return: loss
        """
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
