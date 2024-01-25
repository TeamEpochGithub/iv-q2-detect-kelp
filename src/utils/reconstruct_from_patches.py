"""Reconstruct image from patches."""
import torch


def reconstruct_from_patches(patches: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reconstruct an image from patches.

    :param patches: Tensor of patches.
    :param batch_size: Batch size.
    :return: Reconstructed image.
    """
    if len(patches.shape) == 3:
        patches = patches.unsqueeze(1)
    reconstructed = torch.empty([batch_size, 350, 350], requires_grad=True).to(patches.device)
    # assign the top right and left sides of the image
    # Note that the patched images hold 224x224 images not 350x350
    reconstructed[:, :126, :126] = patches[:batch_size, 0, :126, :126]
    reconstructed[:, :126, -126:] = patches[batch_size : 2 * batch_size, 0, :126, -126:]
    # assign the bottom right and left sides of the image
    reconstructed[:, -126:, :126] = patches[2 * batch_size : 3 * batch_size, 0, -126:, :126]
    reconstructed[:, -126:, -126:] = patches[3 * batch_size : 4 * batch_size, 0, -126:, -126:]
    # assign the top middle
    reconstructed[:, :126, 126:-126] = (patches[:batch_size, 0, :126, -98:] + patches[batch_size : 2 * batch_size, 0, :126, :98]) * 0.5
    # assign the bottom middle
    reconstructed[:, -126:, 126:-126] = (patches[2 * batch_size : 3 * batch_size, 0, -126:, -98:] + patches[3 * batch_size : 4 * batch_size, 0, -126:, :98]) * 0.5
    # assign the middle left
    reconstructed[:, 126:-126, :126] = (patches[:batch_size, 0, -98:, :126] + patches[2 * batch_size : 3 * batch_size, 0, :98, :126]) * 0.5
    # assign the middle right
    reconstructed[:, 126:-126, -126:] = (patches[batch_size : 2 * batch_size, 0, -98:, -126:] + patches[3 * batch_size : 4 * batch_size, 0, :98, -126:]) * 0.5
    # assign the middle
    reconstructed[:, 126:-126, 126:-126] = (
        patches[:batch_size, 0, -98:, -98:]
        + patches[batch_size : 2 * batch_size, 0, -98:, :98]
        + patches[2 * batch_size : 3 * batch_size, 0, :98, -98:]
        + patches[3 * batch_size : 4 * batch_size, 0, :98, :98]
    ) * 0.25
    return reconstructed
