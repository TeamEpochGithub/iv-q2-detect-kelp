"""Auxiliary loss for the double head model, has two losses per head."""
from dataclasses import dataclass

import torch
from segmentation_models_pytorch.losses.dice import DiceLoss
from segmentation_models_pytorch.losses.focal import FocalLoss
from torch import nn

from src.logging_utils.logger import logger


@dataclass
class AuxiliaryLossDouble(nn.Module):
    """AuxiliaryLossDouble class.

    :param classification_weight: Weight for the classification loss
    """

    classification_weight: float = 1.0

    def __post_init__(self) -> None:
        """Initialize the AuxiliaryLoss."""
        super().__init__()
        self.classification_loss_1 = DiceLoss(mode="multiclass")
        self.classification_loss_2 = FocalLoss(mode="multiclass")
        self.regression_loss_1 = torch.nn.MSELoss()
        self.regression_loss_2 = torch.nn.L1Loss()

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

        # Calculate loss for regression and classification and then calculate weighted sum
        regression_loss = self.regression_loss_1(preds[:, :1].contiguous(), targets.contiguous()) + self.regression_loss_2(preds[:, :1].contiguous(), targets.contiguous())
        regression_loss = regression_loss / 2

        classification_loss = self.classification_loss_1(preds[:, 1:], targets.long()) + self.classification_loss_2(preds[:, 1:], targets.long())
        classification_loss = classification_loss / 2

        return (regression_loss + classification_loss * self.classification_weight) / (1 + self.classification_weight)
