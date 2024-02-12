import torch
import numpy as np

def reverse_transform(pred: torch.Tensor, flip:bool, rotation:int) -> torch.Tensor:
    """ Reverse the transformation on the prediction. """
    pred = torch.rot90(pred, -rotation, [2, 3])  # Reverse rotation
    if flip:
        pred = torch.flip(pred, [3])  # Reverse flip
    return pred