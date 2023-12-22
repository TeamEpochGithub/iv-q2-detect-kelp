import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss, also known as soft Sorenson-Dice loss or Tversky loss.

    :param size_average: If True, loss is averaged over the batch. If False, loss is summed over the batch.
    """

    def __init__(self, size_average: bool = True) -> None:
        """Initialize the DiceLoss

        :param size_average: If True, loss is averaged over the batch. If False, loss is summed over the batch.
        """
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> float:
        """Forward pass

        :param inputs: input tensor
        :param targets: target tensor
        :param smooth: smoothing factor
        :return: loss
        """
        # Comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
