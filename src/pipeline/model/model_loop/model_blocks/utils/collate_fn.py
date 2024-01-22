"""Enable getitems to work with the dataloader using this collate function."""


import torch


def collate_fn(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[valid-type]
    """Make getitems work.

    :param batch: The batch of data to collate.
    :return: The collated batch.
    """
    return (batch[0], batch[1])
