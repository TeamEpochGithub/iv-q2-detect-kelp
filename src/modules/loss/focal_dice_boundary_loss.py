"""Focal loss + Dice loss + Boundary loss. Used for training the model."""
import torch
from torch import Tensor, einsum, nn

from src.logging_utils.logger import logger
from src.modules.loss.dice_loss import DiceLoss
from src.modules.loss.focal_loss import FocalLoss


class FocalDiceBoundaryLoss(nn.Module):
    """Focal loss + Dice loss + Boundary loss. Used for training the model."""

    def __init__(self) -> None:
        """Initialize the loss function."""
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = SurfaceLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        """Calculate the loss.

        :param preds: The predicted probabilities.
        :param target: The target mask.
        :return: The loss.
        """
        dist_map = target[:, 1]
        target = target[:, 0]

        return self.focal_loss(preds, target) + self.dice_loss(preds, target) + self.boundary_loss(preds, dist_map) * 0.1


class SurfaceLoss:
    """Boundary loss. Used for training the model."""

    def __init__(self, **kwargs: dict[str, str]) -> None:
        """Initialize the loss function.

        :param kwargs: The parameters for the loss function.
        """
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        logger.debug(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_map: Tensor) -> Tensor:
        """Calculate the boundary loss.

        :param probs: The predicted probabilities.
        :param dist_map: The distance map.
        :return: The boundary loss.
        """
        pc = probs[:, ...].type(torch.float32)
        dc = dist_map[:, ...].type(torch.float32)

        multipled = einsum("bwh,bwh->bwh", pc, dc)

        return multipled.mean()
