"""Padded model architecture, suitable for instantiation for pytorch built-in models."""
import torch
from torch import nn


class PaddedModel(nn.Module):
    """Model architecture with padding.This class is used to wrap a pytorch model and add padding to the input image if necessary.

    :param model: Pytorch model to be used
    :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
    :param activation: activation function to be applied to the output of the model
    """

    def __init__(self, model: nn.Module, padding: int = 0, activation: None | nn.Module = None) -> None:
        """Initialize the PaddedModel.

        :param model: Pytorch model to be used
        :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
        :param activation: activation function to be applied to the output of the model
        """
        super().__init__()
        self.model = model
        self.padding = padding
        self.activation = activation
        # Create a padding layer to pad the input image to a suitable size
        if padding > 0:
            self.padding_layer = nn.ZeroPad2d((padding, padding, padding, padding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        # Pad the input image if necessary
        if self.padding > 0:
            x = self.padding_layer(x)

        # Forward pass
        y = self.model(x)

        # Remove the padding if necessary
        if self.padding > 0:
            if y.ndim == 2:
                y = y[self.padding : -self.padding, self.padding : -self.padding]
            elif y.ndim == 3:
                y = y[:, self.padding : -self.padding, self.padding : -self.padding]
            elif y.ndim == 4:
                y = y[:, :, self.padding : -self.padding, self.padding : -self.padding]
        if self.activation is not None:
            y = self.activation(y)

        return y
