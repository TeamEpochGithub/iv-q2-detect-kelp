import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model


class sSE(nn.Module):
    """Squeeze and Excitation block with spatial squeeze and channel excitation.
    
    :param out_channels: The number of output channels.
    """
    def __init__(self, out_channels: int) -> None:
        """Initialize the sSE block.
        
        :param out_channels: The number of output channels."""
        super(sSE, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(out_channels, 1, kernel_size=1, padding=0),
                                  nn.BatchNorm2d(1),
                                  nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the sSE block.
        
        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.conv(x)
        return x


class cSE(nn.Module):
    """Squeeze and Excitation block with channel squeeze and spatial excitation.
    
    :param out_channels: The number of output channels.
    """

    def __init__(self, out_channels: int) -> None:
        """Initialize the cSE block.
        
        :param out_channels: The number of output channels.
        """
        super(cSE, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(out_channels, int(out_channels/2), kernel_size=1, padding=0),
                                   nn.BatchNorm2d(int(out_channels/2)),
                                   nn.ReLU(inplace=True),
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(int(out_channels/2), out_channels, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(out_channels),
                                   nn.Sigmoid(),
                                   )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the cSE block.

        :param x: The input tensor.
        :return: The output tensor.
        """

        x = nn.AvgPool2d(x.size()[2:])(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MyDecoderBlock(nn.Module):
    """Custom decoder block for the UNet model.
    
    :param in_channel: The number of input channels.
    :param skip_channel: The number of skip channels.
    :param out_channel: The number of output channels.
    """
    def __init__(
        self,
        in_channel: int,
        skip_channel: int,
        out_channel: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel,
                      kernel_size=3, padding=1,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1,),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.spatial_gate = sSE(out_channel)
        self.channel_gate = cSE(out_channel)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the decoder block.
        
        :param x: The input tensor.
        :param skip: The skip tensor.
        :return: The output tensor.
        """
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1*x + g2*x
        return x


class MyUnetDecoder(nn.Module):
    """Custom UNet decoder model.
    
    :param in_channel: The number of input channels.
    :param skip_channel: The number of skip channels.
    :param out_channel: The number of output channels.
    """
    def __init__(self,
                 in_channels: list[int],
                 skip_channels: list[int],
                 out_channels: list[int],
                 ):
        super().__init__()

        self.center = nn.Identity()

        i_channel = [in_channels, ] + out_channels[:-1]
        s_channel = skip_channels
        o_channel = out_channels
        block = [
            MyDecoderBlock(i, s, o,)
            for i, s, o in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature: torch.Tensor, skip: list[torch.Tensor]) -> torch.Tensor: 
        """Forward pass of the UNet decoder.
        
        :param feature: The input tensor.
        :param skip: The skip tensor.
        :return: The output tensor.
        """
        d = self.center(feature)

        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)

        last = d
        return last


class CustomTimm(nn.Module):
    """Custom UNet model based on the timm library.
    
    :param model: The name of the model to use.
    :param in_channels: The number of input channels.
    :param pretrained: Whether to use a pretrained model.
    """

    def __init__(self, model: str, in_channels: int = 3, *, pretrained: bool = True) -> None:
        """Initialize the model.
        
        :param model: The name of the model to use.
        :param in_channels: The number of input channels.
        :param pretrained: Whether to use a pretrained model.
        """
        super().__init__()
        encoder_dim = [24, 48, 96, 192, 384, 768]
        decoder_dim = [384, 192, 96, 48, 24]

        self.encoder = create_model(model, pretrained=pretrained, in_chans=in_channels)

        self.decoder = MyUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1],
            out_channel=decoder_dim,
        )
        self.kelp = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)
        self.stem0 = nn.Sequential(nn.Conv2d(in_channels=7, out_channels=24, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(
                                       24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=24, out_channels=24,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(
                                       24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   )
        self.stem1 = nn.Sequential(nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(
                                       48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=48, out_channels=48,
                                             kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(
                                       48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   )

        self.activation = nn.Sigmoid()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        :param x: The input tensor.
        :return: The output tensor.
        """

        # Pad the input to be divisible by 32
        B, C, H, W = image.shape
        H_pad = (32 - H % 32) % 32
        W_pad = (32 - W % 32) % 32
        x = F.pad(image, (0, W_pad, 0, H_pad), 'constant', 0)

        # Forward pass through the model

        # Encode the input
        encode = []
        xx = self.stem0(x)
        encode.append(xx)
        xx = F.avg_pool2d(xx, kernel_size=2, stride=2)
        xx = self.stem1(xx)
        encode.append(xx)

        e = self.encoder
        x = e.stem(x)

        x = e.stages[0](x)
        encode.append(x)
        x = e.stages[1](x)
        encode.append(x)
        x = e.stages[2](x)
        encode.append(x)
        x = e.stages[3](x)
        encode.append(x)

        # Decode the output
        last = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]
        )

        kelp = self.kelp(last).float()

        kelp = self.activation(kelp)

        kelp = kelp[:, :, :H, :W].contiguous()

        return kelp
