"""Padded model architecture, suitable for instantiation for pytorch built-in models."""
import torch
from torch import nn
from typing import Any

class PaddedModel(nn.Module):
    """Model architecture with padding.This class is used to wrap a pytorch model and add padding to the input image if necessary.

    :param model: Pytorch model to be used
    :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
    :param activation: activation function to be applied to the output of the model
    """

    def __init__(self, model: nn.Module, padding: int = 0, activation: None | nn.Module = None, weights: Any = None) -> None:
        """Initialize the PaddedModel.

        :param model: Pytorch model to be used
        :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
        :param activation: activation function to be applied to the output of the model
        """
        super().__init__()
        self.model = model
        self.padding = padding
        self.activation = activation
        if weights is not None:
            state_dict = weights.get_state_dict(progress=True)
            new_state_dict = self.expand_model_channels(state_dict, default_in_channels=7, new_in_channels=11)
            model.encoder.load_state_dict(new_state_dict)
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
        y = self.model(x).squeeze(axis=1)

        # Remove the padding if necessary
        if self.padding > 0:
            y = y[:, self.padding : -self.padding, self.padding : -self.padding]
        if self.activation is not None:
            y = self.activation(y)

        return y

    def expand_model_channels(self, state_dict, default_in_channels=3, new_in_channels=7):
        for key in list(state_dict.keys()):
            if 'conv' in key and 'weight' in key:
                weight = state_dict[key]
                if weight.size(1) == default_in_channels:
                    new_weight = torch.Tensor(weight.size(0), new_in_channels, *weight.size()[2:])
                    for i in range(new_in_channels):
                        new_weight[:, i] = weight[:, i % default_in_channels]
                    new_weight = new_weight * (default_in_channels / new_in_channels)
                    state_dict[key] = new_weight
        return state_dict