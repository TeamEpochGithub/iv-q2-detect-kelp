"""Auxiliary loss module, has one loss per head."""

import torch
from torch import nn

from src.logging_utils.logger import logger
from src.modules.loss.dice_loss import DiceLoss


class AuxiliaryLoss(nn.Module):
    """AuxiliaryLoss class.

    :param classification_weight: Weight for the classification loss
    """

    def __init__(self, classification_weight: float = 1.0, classification_loss: nn.Module | None = None, regression_loss: nn.Module | None = None) -> None:
        """Initialize the AuxiliaryLoss.

        :param classification_weight: Weight for the classification loss
        :param classification_loss: Loss function for the classification head
        :param regression_loss: Loss function for the regression head
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.classification_loss = nn.BCELoss() if classification_loss is None else classification_loss
        self.regression_loss = DiceLoss() if regression_loss is None else regression_loss

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :return: loss
        """
        # Check inputs shape is correct, last channel should be size 3
        if preds.shape[-3] != 3:
            logger.warning(f"Second last channel should be size 3, got {preds.shape[-3]}")
            raise ValueError(f"Second last channel should be size 3, got {preds.shape[-3]}")

        # One hot encode the targets
        one_hot = torch.nn.functional.one_hot(targets.long().squeeze(), num_classes=2).float()  # (B, H, W, 2)
        one_hot = one_hot.transpose(3, 1)  # (B, 2, H, W)
        targets = torch.cat((targets.unsqueeze(dim=1), one_hot), dim=1)  # (B, 3, H, W)

        # Calculate loss for regression and classification and then calculate weighted sum
        regression_loss = self.regression_loss(preds[:, :1], targets[:, :1])

        classification_loss = self.classification_loss(preds[:, 1:], targets[:, 1:])

        return (regression_loss + classification_loss * self.classification_weight) / (1 + self.classification_weight)
