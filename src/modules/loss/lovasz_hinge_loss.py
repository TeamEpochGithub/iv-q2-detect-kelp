"""Lovasz-Softmax and Jaccard hinge loss in PyTorch.

Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Variable


@dataclass
class LovaszHingeLoss(nn.Module):
    """Implementation of Lovasz hinge loss for image segmentation."""

    def __post_init__(self) -> None:
        """Initialize class."""
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param inputs: input tensor
        :param targets: target tensor
        :return: loss
        """
        return lovasz_hinge(inputs, targets)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors, see Alg. 1 in paper.

    :param gt_sorted: ground truth sorted in descending order
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, ignore: int | None = None) -> torch.Tensor:
    r"""Binary Lovasz hinge loss.

    :param logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
    :param labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    :param ignore: void class id
    """
    return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


def lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    r"""Binary Lovasz hinge loss.

    :param logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
    :param labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(nn.functional.relu(errors_sorted), Variable(grad))


def flatten_binary_scores(scores: torch.Tensor, labels: torch.Tensor, ignore: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Flattens predictions in the batch (binary case).

    Remove labels equal to 'ignore'
    :param scores: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
    :param labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    :param ignore: void class id
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
