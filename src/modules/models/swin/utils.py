"""Utility functions for the Swin Transformer."""
import numpy as np
import torch
from einops import rearrange


def create_mask(window_size: int, displacement: int, *, upper_lower: bool, left_right: bool) -> torch.Tensor:
    """Create a mask for the shifted window attention.

    :param window_size: window size
    :param displacement: displacement
    :param upper_lower: whether to create the upper and lower mask
    :param left_right: whether to create the left and right mask
    :return: mask
    """
    mask = torch.zeros(window_size**2, window_size**2)

    if upper_lower:
        mask[-displacement * window_size :, : -displacement * window_size] = float("-inf")
        mask[: -displacement * window_size, -displacement * window_size :] = float("-inf")

    if left_right:
        mask = rearrange(mask, "(h1 w1) (h2 w2) -> h1 w1 h2 w2", h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float("-inf")
        mask[:, :-displacement, :, -displacement:] = float("-inf")
        mask = rearrange(mask, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")

    return mask


def get_relative_distances(window_size: int) -> torch.Tensor:
    """Get the relative distances for the shifted window attention.

    :param window_size: window size
    :return: relative distances
    """
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    return indices[None, :, :] - indices[:, None, :]
