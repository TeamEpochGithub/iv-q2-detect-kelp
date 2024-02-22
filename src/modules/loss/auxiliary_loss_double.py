"""Auxiliary loss for the double head model, has two losses per head."""

import torch
from segmentation_models_pytorch.losses import DiceLoss as DiceLossSMP
from segmentation_models_pytorch.losses.focal import FocalLoss as FocalLossSMP
from torch import nn


class AuxiliaryLossDouble(nn.Module):
    """AuxiliaryLossDouble class.

    :param classification_weight: Weight for the classification loss
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        classification_loss_1: nn.Module | None = None,
        classification_loss_2: nn.Module | None = None,
        regression_loss_1: nn.Module | None = None,
        regression_loss_2: nn.Module | None = None,
    ) -> None:
        """Initialize the AuxiliaryLossDouble.

        :param classification_weight: Weight for the classification loss
        :param classification_loss_1: Loss function for the first classification head
        :param classification_loss_2: Loss function for the second classification head
        :param regression_loss_1: Loss function for the first regression head
        :param regression_loss_2: Loss function for the second regression head
        """
        super().__init__()
        self.classification_weight = classification_weight
        self.classification_loss_1 = DiceLossSMP(mode="multiclass") if classification_loss_1 is None else classification_loss_1
        self.classification_loss_2 = FocalLossSMP(mode="multiclass") if classification_loss_2 is None else classification_loss_2
        self.regression_loss_1 = torch.nn.MSELoss() if regression_loss_1 is None else regression_loss_1
        self.regression_loss_2 = torch.nn.L1Loss() if regression_loss_2 is None else regression_loss_2

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :return: loss
        """
        # Calculate loss for regression and classification and then calculate weighted sum
        regression_loss = self.regression_loss_1(preds[:, :1].contiguous(), targets.contiguous()) + self.regression_loss_2(preds[:, :1].contiguous(), targets.contiguous())
        regression_loss = regression_loss / 2

        classification_loss = self.classification_loss_1(preds[:, 1:].contiguous(), targets.long()) + self.classification_loss_2(preds[:, 1:].contiguous(), targets.long())
        classification_loss = classification_loss / 2

        return (regression_loss + classification_loss * self.classification_weight) / (1 + self.classification_weight)
