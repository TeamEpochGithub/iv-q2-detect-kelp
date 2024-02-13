import torch.nn as nn
from torchvision.models.convnext import convnext_tiny
import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
import segmentation_models_pytorch as smp
from timm.models import create_model

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    
)

import segmentation_models_pytorch as smp
    
class ConvNext(nn.Module):

    def __init__(self, classes: int = 1, in_channels: int = 3, activation: str = 'sigmoid') -> None:
        super().__init__()
        self.model = SegmentationModel('convnext_tiny.fb_in1k', in_channels=in_channels, classes=classes)


    def forward(self, x: torch.tensor) -> torch.tensor:

        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:] # (96, 192, 384, 768)
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1] # (768, 384, 192, 96)

        # computing blocks input and output channels
        head_channels = encoder_channels[0] # 768
        in_channels = [head_channels] + list(decoder_channels[:-1]) # (768, 384, 192, 96)
        skip_channels = list(encoder_channels[1:]) + [0] # (384, 192, 96, 0)
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        out = []
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            out.append(x)

        return out[::-1]
    

class SegmentationModel(nn.Module):
    def __init__(self, 
                encoder,
                encoder_weights=None,
                encoder_depth=4,
                in_channels=3,
                decoder_use_batchnorm: bool = True,
                decoder_channels = (384, 192, 96, 7),
                decoder_attention_type = None,
                classes = 1
                 ):
        super().__init__()
        self.encoder = create_model(model_name=encoder,
                             in_chans=in_channels,
                             pretrained=(encoder_weights is not None),
                             features_only=True)

        self.decoder = UnetDecoder(
            encoder_channels=(7, 96, 192, 384, 768),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = nn.ModuleList()
        for channel in decoder_channels[::-1]:
            self.segmentation_head.append(
                SegmentationHead(
                    in_channels=channel,
                    out_channels=classes,
                    activation=None,
                    kernel_size=3,
                )
            )

    def forward(self,x):

        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        
        masks = []
        for i,seg_head in enumerate(self.segmentation_head):
            masks.append(seg_head(decoder_output[i]))

        return masks