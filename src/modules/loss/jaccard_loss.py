"""Implementation of Jaccard for image segmentation."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class JaccardLoss(nn.Module):
    """Implementation of Jaccard loss for image segmentation."""

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
        # Flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        # Intersection is equivalent to True Positive count
        # Union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        jaccard = (intersection + smooth) / (union + smooth)

        return 1 - jaccard
