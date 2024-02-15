from scipy.ndimage import distance_transform_edt
from torch import Tensor, einsum
import torch
from src.modules.loss.dice_loss import DiceLoss
from src.modules.loss.focal_loss import FocalLoss

from src.modules.loss.utils import simplex, probs2one_hot, one_hot
from src.modules.loss.utils import one_hot2hd_dist

from torch import nn

class FocalDiceBoundaryLoss(nn.Module):

    def __init__(self, alpha=0.5, gamma=2, beta=1):
        super().__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = SurfaceLoss()

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:


        dist_map = target[:, 1]
        target = target[:, 0]

        loss = self.focal_loss(preds, target) + self.dice_loss(preds, target) + self.boundary_loss(preds, dist_map) * 0.1

        return loss
        

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_map: Tensor) -> Tensor:

        pc = probs[:, ...].type(torch.float32)
        dc = dist_map[:, ...].type(torch.float32)

        multipled = einsum("bwh,bwh->bwh", pc, dc)

        loss = multipled.mean()

        return loss