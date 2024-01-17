"""Implementation of Focal Loss."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced data."""

    alpha: float = 0.25
    gamma: float = 2.0
    reduction: str = "mean"

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param inputs: Predictions from the model after sigmoid activation (probabilities between 0 and 1)
        :param targets: Ground truth labels
        """
        # BCE Loss
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")

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
            raise ValueError(f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
        return loss
