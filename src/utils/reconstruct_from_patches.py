"""Reconstruct image from patches."""
import torch


def reconstruct_from_patches(patches: torch.Tensor, batch_size: int, patch_size: int = 256, output_size: int = 350) -> torch.Tensor:
    """Reconstruct an image from patches.

    :param patches: Tensor of patches.
    :param batch_size: Batch size.
    :return: Reconstructed image.
    """
    if len(patches.shape) == 3:
        patches = patches.unsqueeze(1)
    reconstructed = torch.empty([batch_size, output_size, output_size]).to(patches.device)
    # Assign the top right and left sides of the image
    # Note that the patched images hold 224x224 images not 350x350
    idx1 = output_size - patch_size
    idx2 = 2 * patch_size - output_size

    reconstructed[:, :idx1, :idx1] = patches[:batch_size, 0, :idx1, :idx1]
    reconstructed[:, :idx1, -idx1:] = patches[batch_size : 2 * batch_size, 0, :idx1, -idx1:]
    # Assign the bottom right and left sides of the image
    reconstructed[:, -idx1:, :idx1] = patches[2 * batch_size : 3 * batch_size, 0, -idx1:, :idx1]
    reconstructed[:, -idx1:, -idx1:] = patches[3 * batch_size : 4 * batch_size, 0, -idx1:, -idx1:]
    # Assign the top middle
    reconstructed[:, :idx1, idx1:-idx1] = (patches[:batch_size, 0, :idx1, -idx2:] + patches[batch_size : 2 * batch_size, 0, :idx1, :idx2]) * 0.5
    # Assign the bottom middle
    reconstructed[:, -idx1:, idx1:-idx1] = (patches[2 * batch_size : 3 * batch_size, 0, -idx1:, -idx2:] + patches[3 * batch_size : 4 * batch_size, 0, -idx1:, :idx2]) * 0.5
    # Assign the middle left
    reconstructed[:, idx1:-idx1, :idx1] = (patches[:batch_size, 0, -idx2:, :idx1] + patches[2 * batch_size : 3 * batch_size, 0, :idx2, :idx1]) * 0.5
    # Assign the middle right
    reconstructed[:, idx1:-idx1, -idx1:] = (patches[batch_size : 2 * batch_size, 0, -idx2:, -idx1:] + patches[3 * batch_size : 4 * batch_size, 0, :idx2, -idx1:]) * 0.5
    # Assign the middle
    reconstructed[:, idx1:-idx1, idx1:-idx1] = (
        patches[:batch_size, 0, -idx2:, -idx2:]
        + patches[batch_size : 2 * batch_size, 0, -idx2:, :idx2]
        + patches[2 * batch_size : 3 * batch_size, 0, :idx2, -idx2:]
        + patches[3 * batch_size : 4 * batch_size, 0, :idx2, :idx2]
    ) * 0.25
    return reconstructed.unsqueeze(1)
