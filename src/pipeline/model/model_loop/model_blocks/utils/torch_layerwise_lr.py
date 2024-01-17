"""Utilities for setting layerwise learning rates in PyTorch."""


from typing import Any

from torch import nn

from src.logging_utils.logger import logger


def torch_layerwise_lr_groups(model: nn.Module, base_lr: float, layerwise_lr_decay: float, params_per_layer: int = 8) -> list[dict[str, Any]]:
    """Create the parameter groups for the optimizer.

    Groups the parameters individually and sets the learning rate for each group.
    Treats every parameter as a layer, and assumes they are ordered from first layer to last layer.
    Learning rate is highest for the last layer and lowest for the first layer.
    See: https://gist.github.com/gautierdag/3bd64f33470cb11f4323ce7fa86524a9

    :param model: Model to set the learning rate for.
    :param base_lr: Base learning rate.
    :param layerwise_lr_decay: Decay factor for the learning rate.
    :param params_per_layer: Since the architecture is not known, we assume each layer has this many parameters.
    :return: Parameter groups, a list of dictionaries with parameters and learning rate.
    """
    # Set the learning rate for each layer
    num_params = len(list(model.parameters()))
    grouped_params = []
    for i, (_, params) in enumerate(model.named_parameters()):
        depth = (num_params - i - 1) // params_per_layer  # depth will be 0 for the last layer
        lr = base_lr * (layerwise_lr_decay**depth)
        grouped_params.append({"params": params, "lr": lr})

    lowest_lr = grouped_params[0]["lr"]
    highest_lr = grouped_params[-1]["lr"]
    logger.info(f"Set learning rates with layer decay from {lowest_lr} to {highest_lr}")
    return grouped_params
