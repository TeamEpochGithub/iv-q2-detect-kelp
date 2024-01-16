"""Implementation of Shrinkage Loss."""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ShrinkageLoss(nn.Module):
    """Shrinkage Loss.

    :param reduction: Reduction mode for the loss. Options are 'mean' and 'sum'.
    """
    reduction: str = "mean"

    def __post_init__(self):
        super(ShrinkageLoss, self).__init__()

    def forward(self, inputs, targets, alpha: int = 2, c: int = 0.2):
        l1_loss = torch.abs(inputs - targets)
        shrinkage_loss = (l1_loss ** 2) * torch.exp(targets) / \
                         (1 + torch.exp(alpha * (c - l1_loss)))
        if self.reduction == "mean":
            return shrinkage_loss.mean()
        return shrinkage_loss
