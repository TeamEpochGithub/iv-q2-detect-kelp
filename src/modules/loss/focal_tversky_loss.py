"""Implementation of FocalTversky for image segmentation."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FocalTverskyLoss(nn.Module):
    """Implementation of FocalTversky loss for image segmentation. """
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 1.0

    def __post_init__(self):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :param smooth: smoothing factor
        :return: loss
        """
        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky