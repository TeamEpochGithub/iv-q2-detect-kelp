"""Implementation of Dice Loss + BCE for image segmentation."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DiceBCELoss(nn.Module):
    """Dice BCE loss

    :param threshold: threshold for converting predictions to binary values
    """
    threshold: float = -1

    def __post_init__(self):
        super(DiceBCELoss, self).__init__()

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
        dice = 1 - ((2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth))
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return dice + bce
