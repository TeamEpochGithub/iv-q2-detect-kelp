"""This will import the unet architecture from segmentation models pytorch and add padding to the input image to make it divisible by 32"""
from segmentation_models_pytorch import Unet
from torch import nn


class Unet_w_padding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = Unet(in_channels=in_channels, classes=out_channels, activation="sigmoid")
        # create a padding layer to pad the input image from 350x350 to 352x352
        self.padding = nn.ZeroPad2d((1, 1, 1, 1))

    def forward(self, x):
        # pad the input image
        x_padded = self.padding(x)
        # pass the padded image through the model
        y_padded = self.model(x_padded).squeeze()
        # remove the padding
        y = y_padded[:, 1:-1, 1:-1]
        return y
