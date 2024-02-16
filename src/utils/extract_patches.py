"""Make a list of patches from an image."""
import torch


def extract_patches(img: torch.Tensor, patch_size: int = 256) -> torch.Tensor:
    """Extract patches from an image."""
    # Initialize a list to store the patches
    patches = []

    # Extract the patches
    patches.append(img[:, :, :patch_size, :patch_size])  # Top-left corner
    patches.append(img[:, :, :patch_size, -patch_size:])  # Top-right corner
    patches.append(img[:, :, -patch_size:, :patch_size])  # Bottom-left corner
    patches.append(img[:, :, -patch_size:, -patch_size:])  # Bottom-right corner

    # Stack the patches into a tensor along the batch dimension
    return torch.cat(patches, dim=0)
