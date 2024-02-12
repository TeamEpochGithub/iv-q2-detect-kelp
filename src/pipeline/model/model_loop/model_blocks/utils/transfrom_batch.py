import torch


def transform_batch(X: torch.Tensor, flip: bool, rotation: int) -> torch.Tensor:
    """Apply flip and rotation transformations to the batch."""
    if flip:
        X = torch.flip(X, [3])  # Flip horizontally
    X = torch.rot90(X, rotation, [2, 3])  # Rotate
    return X
