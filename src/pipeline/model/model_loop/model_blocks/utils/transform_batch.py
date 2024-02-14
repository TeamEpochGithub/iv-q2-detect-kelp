"""Apply flip and rotation transformations to the batch."""
import torch


def transform_batch(X: torch.Tensor, rotation: int, *, flip: bool) -> torch.Tensor:
    """Apply flip and rotation transformations to the batch."""
    if flip:
        X = torch.flip(X, [3])  # Flip horizontally
    return torch.rot90(X, rotation, [2, 3])  # Rotate
