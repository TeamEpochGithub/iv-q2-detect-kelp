"""Implementation of ComboLoss for image segmentation.

Combo loss is a combination of Dice loss and Cross-entropy loss. Penalises false positives or false negatives more than the other.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ComboLoss(nn.Module):
    """Implementation of Combo loss for image segmentation."""

    alpha: float = 0.5  # < 0.5 penalises FP more, > 0.5 penalises FN more
    ce_ratio: float = 0.5  # Weighted contribution of modified CE loss compared to Dice loss

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1, epsilon: float = 1e-9) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :param smooth: smoothing factor
        :param epsilon: epsilon value
        :return: loss
        """
        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Calculate Dice score
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, epsilon, 1.0 - epsilon)
        out = -(self.alpha * ((targets * torch.log(inputs + epsilon)) + ((1 - self.alpha) * (1.0 - targets) * torch.log(1.0 - inputs + epsilon))))
        weighted_ce = torch.mean(out)
        return (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
