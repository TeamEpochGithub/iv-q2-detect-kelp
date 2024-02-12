"""Patch merging module."""
import torch
from torch import nn


class PatchMerging(nn.Module):
    """Patch merging module.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param downscaling_factor: factor by which the input image is downscaled
    """

    def __init__(self, in_channels: int, out_channels: int, downscaling_factor: int) -> None:
        """Initialize the PatchMerging module.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param downscaling_factor: factor by which the input image is downscaled
        """
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor**2, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        # Unfold the input tensor
        b, c, h, w = x.shape

        # Set the new height and width
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor

        # Apply the patch merging
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        return self.linear(x)
