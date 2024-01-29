from torch import nn
from dataclasses import dataclass

import torch

from src.modules.loss.dice_loss import DiceLoss


@dataclass
class AuxiliaryLoss(nn.Module): 

    classification_weight: float = 1.0

    def __post_init__(self) -> None:
        """Initialize the AuxiliaryLoss."""
        super().__init__()
        self.classification_loss = nn.BCELoss()
        self.regression_loss = DiceLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        :param inputs: input tensor
        :param targets: target tensor
        :return: loss
        """

        # Check inputs shape is correct, last channel should be size 3
        assert preds.shape[-3] == 3, f"Second last channel should be size 3, got {preds.shape[-3]}"

        # One hot encode the targets
        one_hot = torch.nn.functional.one_hot(targets.long().squeeze(), num_classes=2).float() # (B, H, W, 2)
        one_hot = one_hot.transpose(3, 1) # (B, 2, H, W)
        targets = torch.cat((targets.unsqueeze(dim=1), one_hot), dim=1) # (B, 3, H, W)

        # Calculate loss for regression and classification and then calculate weighted sum
        regression_loss = self.regression_loss(preds[:, :1], targets[:, :1])

        classification_loss = self.classification_loss(preds[:, 1:], targets[:, 1:])

        loss = (regression_loss + classification_loss * self.classification_weight) / (1 + self.classification_weight)

        return loss