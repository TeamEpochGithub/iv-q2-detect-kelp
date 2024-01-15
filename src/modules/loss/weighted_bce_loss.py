"""Implementation of Weighted Binary Cross Entropy Loss."""

import torch
from torch import nn

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss.

    :param size_average: If True, loss is averaged over the batch. If False, loss is summed over the batch.
    """

    def __init__(self, positive_freq: float) -> None:
        """Initialize the WeightedBCELoss."""
        super().__init__()
        self.positive_freq = positive_freq

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epsilon = 1e-7) -> float:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :param weight: weight tensor
        :return: loss
        """
        loss_pos = -1 * torch.mean((1 - self.positive_freq) * (targets * torch.log(inputs + epsilon)))
        loss_neg = -1 * torch.mean(self.positive_freq * (1 - targets) * torch.log((1 - inputs) + epsilon))
        return loss_neg + loss_pos

