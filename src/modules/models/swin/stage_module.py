"""Stage module for Swin Transformer."""
import torch
from torch import nn

from src.modules.models.swin.patch_merging import PatchMerging
from src.modules.models.swin.swin_block import SwinBlock


class StageModule(nn.Module):
    """Stage module for Swin Transformer.

    :param in_channels: number of input channels
    :param hidden_dimension: hidden dimension
    :param layers: number of layers
    :param downscaling_factor: factor by which the input image is downscaled
    :param num_heads: number of heads
    :param head_dim: head dimension
    :param window_size: window size
    :param relative_pos_embedding: whether to use relative positional embeddings
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dimension: int,
        layers: int,
        downscaling_factor: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        *,
        relative_pos_embedding: bool = False,
    ) -> None:
        """Initialize the StageModule.

        :param in_channels: number of input channels
        :param hidden_dimension: hidden dimension
        :param layers: number of layers
        :param downscaling_factor: factor by which the input image is downscaled
        :param num_heads: number of heads
        :param head_dim: head dimension
        :param window_size: window size
        :param relative_pos_embedding: whether to use relative positional embeddings
        """
        super().__init__()
        if layers % 2 != 0:
            raise ValueError("Stage layers need to be divisible by 2 for regular and shifted block.")

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension, downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(
                nn.ModuleList(
                    [
                        SwinBlock(
                            dim=hidden_dimension,
                            heads=num_heads,
                            head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4,
                            shifted=False,
                            window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding,
                        ),
                        SwinBlock(
                            dim=hidden_dimension,
                            heads=num_heads,
                            head_dim=head_dim,
                            mlp_dim=hidden_dimension * 4,
                            shifted=True,
                            window_size=window_size,
                            relative_pos_embedding=relative_pos_embedding,
                        ),
                    ],
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)
