"""Swin Transformer block."""
import torch
from torch import nn

from src.modules.models.swin.feed_forward import FeedForward
from src.modules.models.swin.pre_norm import PreNorm
from src.modules.models.swin.residual import Residual
from src.modules.models.swin.window_attention import WindowAttention


class SwinBlock(nn.Module):
    """Swin Transformer block.

    :param dim: input dimension
    :param heads: number of heads
    :param head_dim: head dimension
    :param mlp_dim: dimension of the feedforward network
    :param shifted: whether to use shifted window attention
    :param window_size: window size
    :param relative_pos_embedding: whether to use relative positional embeddings
    """

    def __init__(self, dim: int, heads: int, head_dim: int, mlp_dim: int, window_size: int, *, shifted: bool, relative_pos_embedding: bool = False) -> None:
        """Initialize the SwinBlock.

        :param dim: input dimension
        :param heads: number of heads
        :param head_dim: head dimension
        :param mlp_dim: dimension of the feedforward network
        :param shifted: whether to use shifted window attention
        :param window_size: window size
        :param relative_pos_embedding: whether to use relative positional embeddings
        """
        super().__init__()
        self.attention_block = Residual(
            PreNorm(dim, WindowAttention(dim=dim, heads=heads, head_dim=head_dim, shifted=shifted, window_size=window_size, relative_pos_embedding=relative_pos_embedding)),
        )
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        x = self.attention_block(x)
        return self.mlp_block(x)
