"""Reconstruct image from patches."""
import torch


def reconstruct_from_patches(patches: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reconstruct an image from patches.

    :param patches: Tensor of patches.
    :param batch_size: Batch size.
    :return: Reconstructed image.
    """
    reconstructed = torch.zeros([batch_size, 350, 350])
    patches = torch.reshape(patches, (batch_size, 4, 224, 224))
    reconstructed[:, :224, :224] += patches[:, 0, :, :]
    reconstructed[:, :224, -224:] += patches[:, 1, :, :]
    reconstructed[:, -224:, :224] += patches[:, 2, :, :]
    reconstructed[:, -224:, -224:] += patches[:, 3, :, ::]
    reconstructed[:, 126:224, :] = reconstructed[:, 126:224, :] * 0.5
    reconstructed[:, :, 126:224] = reconstructed[:, :, 126:224] * 0.5
    return reconstructed
