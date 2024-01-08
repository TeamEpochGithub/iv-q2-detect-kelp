"""Imports the unet architecture from segmentation models pytorch and adds padding to the input image to make it divisible by 32."""
import torch
from segmentation_models_pytorch import Unet
from torch import nn


class PaddedUnet(nn.Module):
    """Unet architecture with padding to make the input divisible by 32.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the PaddedUnet.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        super().__init__()
        self.model = Unet(in_channels=in_channels, classes=out_channels, activation="sigmoid")
        # create a padding layer to pad the input image from 350x350 to 352x352
        self.padding = nn.ZeroPad2d((1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        # pad the input image
        x_padded = self.padding(x)
        # pass the padded image through the model
        y_padded = self.model(x_padded).squeeze(axis=1)
        # remove the padding and return
        return y_padded[:, 1:-1, 1:-1]
