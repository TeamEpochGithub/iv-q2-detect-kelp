"""Padded model architecture, suitable for instantiation for pytorch built-in models."""
import torch
from torch import nn

from src.utils.extract_patches import extract_patches
from src.utils.reconstruct_from_patches import reconstruct_from_patches


class GridModel(nn.Module):
    """Model architecture with padding.This class is used to wrap a pytorch model and add padding to the input image if necessary.

    :param model: Pytorch model to be used
    :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
    :param activation: activation function to be applied to the output of the model
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the PaddedModel.

        :param model: Pytorch model to be used
        :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
        :param activation: activation function to be applied to the output of the model
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        # Create patches from the input image
        patches = extract_patches(x)
        # Forward pass
        y = self.model(patches).squeeze(axis=1)
        # Reconstruct the image from the patches
        return reconstruct_from_patches(y, x.shape[0])
