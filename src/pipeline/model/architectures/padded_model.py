"""Padded model architecture, suitable for instantiation for pytorch built-in models."""
import torch
from torch import nn


class PaddedModel(nn.Module):
    """Model architecture with padding.This class is used to wrap a pytorch model and add padding to the input image if necessary.

    :param model: Pytorch model to be used
    :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
    """

    def __init__(self, model: nn.Module, padding: int = 0) -> None:
        """Initialize the PaddedModel.

        :param model: Pytorch model to be used
        :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
        """
        super().__init__()
        self.model = model
        self.padding = padding
        # Create a padding layer to pad the input image to a suitable size
        if padding > 0:
            self.padding_layer = nn.ZeroPad2d((padding, padding, padding, padding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        # Pad the input image if necessary
        if hasattr(self, "padding"):
            x = self.padding_layer(x)

        # Forward pass
        y = self.model(x).squeeze(axis=1)

        # Remove the padding if necessary
        if hasattr(self, "padding"):
            y = y[:, self.padding : -self.padding, self.padding : -self.padding]

        return y
