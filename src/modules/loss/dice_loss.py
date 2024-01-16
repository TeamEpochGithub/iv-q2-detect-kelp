"""Implementation of Dice Loss for image segmentation."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiceLoss(nn.Module):
    """Dice loss, also known as soft Sorenson-Dice loss or Tversky loss.

    :param threshold: threshold for converting predictions to binary values
    """
    threshold: float = -1

    def __post_init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :param smooth: smoothing factor
        :return: loss
        """
        # Apply threshold if not -1
        if self.threshold != -1:
            inputs = torch.where(inputs > self.threshold, 1, 0)

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
