"""Window attention layer."""
import torch
from einops import rearrange
from torch import einsum, nn

from src.modules.models.swin.cyclic_shift import CyclicShift
from src.modules.models.swin.utils import create_mask, get_relative_distances


class WindowAttention(nn.Module):
    """Window attention layer.

    :param dim: input dimension
    :param heads: number of heads
    :param head_dim: head dimension
    :param shifted: whether to use shifted window attention
    :param window_size: window size
    :param relative_pos_embedding: whether to use relative positional embeddings
    """

    def __init__(self, dim: int, heads: int, head_dim: int, window_size: int, *, shifted: bool, relative_pos_embedding: bool) -> None:
        """Initialize the WindowAttention.

        :param dim: input dimension
        :param heads: number of heads
        :param head_dim: head dimension
        :param shifted: whether to use shifted window attention
        :param window_size: window size
        :param relative_pos_embedding: whether to use relative positional embeddings
        """
        super().__init__()

        # Inner dimension
        inner_dim = head_dim * heads

        # Store the parameters
        self.heads = heads
        self.scale = head_dim**-0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        # If shifted, create the cyclic shift layer and the masks
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=False, left_right=True), requires_grad=False)

        # Linear layers
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Positional embedding
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :return: output tensor
        """
        # Apply the cyclic shift if needed
        if self.shifted:
            x = self.cyclic_shift(x)

        # Get the shape of the input tensor
        _, n_h, n_w, _, h = *x.shape, self.heads

        # Linear transformation
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(lambda t: rearrange(t, "b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d", h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        # Add the positional embedding
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        # Add the masks if needed
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1 :: nw_w] += self.left_right_mask

        # Get the attention
        attn = dots.softmax(dim=-1)

        # Get the output
        out = einsum("b h w i j, b h w j d -> b h w i d", attn, v)
        out = rearrange(out, "b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)", h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        # Apply the cyclic back shift if needed
        if self.shifted:
            out = self.cyclic_back_shift(out)

        return out
