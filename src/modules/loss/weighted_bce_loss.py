"""Implementation of Weighted Binary Cross Entropy Loss."""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss.

    :param positive_freq: Frequency of positive examples in the dataset.
    :param size_average: If True, loss is averaged over the batch. If False, loss is summed.
    """

    positive_freq: float = 0.5
    size_average: bool = True

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :return: loss
        """
        loss_pos = -1 * (1 - self.positive_freq) * (targets * torch.log(inputs + epsilon))
        loss_neg = -1 * self.positive_freq * (1 - targets) * torch.log((1 - inputs) + epsilon)
        loss = loss_pos + loss_neg
        return torch.mean(loss) if self.size_average else torch.sum(loss)
