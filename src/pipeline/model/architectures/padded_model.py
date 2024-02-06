"""Padded model architecture, suitable for instantiation for pytorch built-in models."""
from collections.abc import MutableMapping
from typing import Any

import torch
import torchvision
from torch import nn


class PaddedModel(nn.Module):
    """Wrapper for pretrained `SMP <https://github.com/qubvel/segmentation_models.pytorch>`_ models.

    Use this class to load pretrained weights into SMP models and add padding if necessary.
    """

    def __init__(
        self,
        model: nn.Module,
        padding: int = 0,
        activation: nn.Module | None = None,
        pretrained_weights: torchvision.models.WeightsEnum | None = None,
        default_in_channels: int = 7,
        new_in_channels: int = 11,
    ) -> None:
        """Load the pretrained weights.

        :param model: SMP model to be used.
        :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images).
        :param activation: activation function to be applied to the output of the model.
        :param pretrained_weights: pretrained weights to be loaded into the model.
        :param default_in_channels: default number of input channels.
        :param new_in_channels: new number of input channels.
        """
        super().__init__()
        self.model = model
        self.padding = padding
        self.activation = activation

        if pretrained_weights is not None:
            state_dict = pretrained_weights.get_state_dict(progress=True)
            new_state_dict = expand_model_channels(state_dict, default_in_channels=default_in_channels, new_in_channels=new_in_channels)
            self.model.encoder.load_state_dict(new_state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        # Pad the input image if necessary
        if self.padding > 0:
            x = nn.ZeroPad2d(self.padding)(x)

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


def expand_model_channels(state_dict: MutableMapping[str, Any], default_in_channels: int = 3, new_in_channels: int = 7) -> MutableMapping[str, Any]:
    """Expand the number of input channels of the model's encoder.

    :param state_dict: state dict of the model
    :param default_in_channels: default number of input channels
    :param new_in_channels: new number of input channels
    :return: new state dict with expanded number of input channels
    """
    for key in state_dict:
        if "conv" in key and "weight" in key:
            weight = state_dict[key]
            if weight.size(1) == default_in_channels:
                new_weight = torch.Tensor(weight.size(0), new_in_channels, *weight.size()[2:])
                for i in range(new_in_channels):
                    new_weight[:, i] = weight[:, i % default_in_channels]
                new_weight = new_weight * (default_in_channels / new_in_channels)
                state_dict[key] = new_weight
    return state_dict
