import torch
from src.utils.extract_patches import extract_patches
from src.utils.reconstruct_from_patches import reconstruct_from_patches

from unittest import TestCase
class TestShore(TestCase):
    def test_extract_patches(self):
        example_model = torch.nn.Conv2d(6, 1, 1)
        # Set the weights to 1 divided by the number of input channels
        example_model.weight.data.fill_(1. / 6)
        # Set the bias to 0
        example_model.bias.data.fill_(0.)
        test_batch = torch.ones([32,6,350,350])
        with torch.no_grad():
            example_output = example_model(test_batch)
        patches = extract_patches(example_output)
        reconstructed = reconstruct_from_patches(patches, 32)
        example_output = torch.ones([32,350,350])
        assert torch.all(reconstructed == example_output)