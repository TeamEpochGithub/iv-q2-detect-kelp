"""Model heads."""
import torch
from torch import nn


class RegressionHead(nn.Module):
    """Regression head for the model.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: kernel size for the convolutional layers
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        """Initialize the RegressionHead.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size for the convolutional layers
        """
        super().__init__()
        self.conv2d0 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=1)
        self.activation0 = nn.ReLU(inplace=True)
        self.conv2d1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=kernel_size, padding=1)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2d2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.activation2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        x = self.conv2d0(x)
        x = self.activation0(x)
        x = self.conv2d1(x)
        x = self.activation1(x)
        x = self.conv2d2(x)
        return self.activation2(x)


class SegmentationHead(nn.Module):
    """Segmentation head for the model.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: kernel size for the convolutional layers
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        """Initialize the SegmentationHead.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size for the convolutional layers
        """
        super().__init__()
        self.conv2d0 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=1)
        self.activation0 = nn.ReLU(inplace=True)
        self.conv2d1 = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=kernel_size, padding=1)
        self.activation1 = nn.ReLU(inplace=True)
        self.conv2d2 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.upsample = nn.Upsample(size=(), mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        x = self.conv2d0(x)
        x = self.activation0(x)
        x = self.conv2d1(x)
        x = self.activation1(x)
        return self.conv2d2(x)
