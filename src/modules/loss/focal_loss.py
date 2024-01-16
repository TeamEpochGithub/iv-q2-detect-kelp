"""Implementation of Focal Loss."""
from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

@dataclass
class FocalLoss(nn.Module):
    alpha: float = 0.25
    gamma: float = 2.0
    reduction: str = "mean"

    def __post_init__(self):
        super(FocalLoss, self).__init__()


    def forward(self, inputs, targets):
        """
        :param inputs: Predictions from the model after sigmoid activation (probabilities between 0 and 1)
        :param targets: Ground truth labels
        """
        # BCE Loss
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Calculate focal loss
        p = inputs
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = BCE_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
