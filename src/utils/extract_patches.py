"""Make a list of patches from an image."""
import torch


def extract_patches(img: torch.Tensor) -> torch.Tensor:
    """Extract patches from an image."""
    # Initialize a list to store the patches
    patches = []

    # Extract the patches
    patches.append(img[:, :, :224, :224])  # Top-left corner
    patches.append(img[:, :, :224, -224:])  # Top-right corner
    patches.append(img[:, :, -224:, :224])  # Bottom-left corner
    patches.append(img[:, :, -224:, -224:])  # Bottom-right corner

    # Stack the patches into a tensor along the batch dimension
    return torch.cat(patches, dim=0)
