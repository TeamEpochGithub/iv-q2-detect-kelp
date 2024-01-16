"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


@dataclass
class LovaszHingeLoss(nn.Module):
    """Implementation of Lovasz hinge loss for image segmentation. """

    per_image: bool = False

    def __post_init__(self):
        """Initialize class."""
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :return: loss
        """
        loss = lovasz_hinge(inputs, targets, self.per_image)
        return loss


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, per_image: bool = False, ignore: int | None = None) -> torch.Tensor:
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores: torch.Tensor, labels: torch.Tensor, ignore=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'

    :param scores: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
    :param labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    :param ignore: void class id
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def isnan(x: torch.Tensor) -> bool:
    """ Returns true if x is nan.

    :param x: tensor
    """

    return x != x


def mean(l: torch.Generator, ignore_nan: bool = False, empty: str | int = 0) -> torch.Tensor:
    """
    nanmean compatible with generators.

    :param l: generator
    :param ignore_nan: ignore nan values
    :param empty: empty value
    """

    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
