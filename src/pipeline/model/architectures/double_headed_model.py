"""Padded model architecture, suitable for instantiation for pytorch built-in models."""
import torch
from torch import nn

from src.modules.models.model_head import RegressionHead, SegmentationHead


class DoubleHeadedModel(nn.Module):
    """Model architecture for the double headed model.This class is used to wrap a pytorch model and add padding to the input image if necessary.

    :param model: Pytorch model to be used
    :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
    """

    def __init__(self, model: nn.Module, padding: int = 0) -> None:
        """Initialize the PaddedModel.

        :param model: Pytorch model to be used
        :param padding: padding to be applied to the input image (to allow, for example, Unet to work with 350x350 images)
        :param activation: activation function to be applied to the output of the model
        """
        super().__init__()
        self.model = model
        self.padding = padding
        # Create a padding layer to pad the input image to a suitable size
        if padding > 0:
            self.padding_layer = nn.ZeroPad2d((padding, padding, padding, padding))

        # Replace the last layer with a new one
        if hasattr(self.model, "segmentation_head"):
            self.model.segmentation_head = nn.Identity()

        self.segmentation_head = SegmentationHead(16, 2)
        self.regression_head = RegressionHead(16, 1)
        self.remove_head = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tuple of tensors
        """
        # Pad the input image if necessary
        if self.padding > 0:
            x = self.padding_layer(x)

        # Forward pass
        y = self.model(x)

        if self.remove_head:
            return y

        # Apply the heads
        y_seg = self.segmentation_head(y)
        y_reg = self.regression_head(y)

        # Remove the padding if necessary from both outputs
        if self.padding > 0:
            if y_seg.ndim == 2:
                y_seg = y_seg[self.padding : -self.padding, self.padding : -self.padding]
                y_reg = y_reg[self.padding : -self.padding, self.padding : -self.padding]
            elif y_seg.ndim == 3:
                y_seg = y_seg[:, self.padding : -self.padding, self.padding : -self.padding]
                y_reg = y_reg[:, self.padding : -self.padding, self.padding : -self.padding]
            elif y_seg.ndim == 4:
                y_seg = y_seg[:, :, self.padding : -self.padding, self.padding : -self.padding]
                y_reg = y_reg[:, :, self.padding : -self.padding, self.padding : -self.padding]

        return y_seg, y_reg

    def remove_heads(self) -> None:
        """Remove the heads from the model."""
        self.remove_head = True
