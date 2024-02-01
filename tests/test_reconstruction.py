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
        test_batch = torch.rand([32,1,350,350],requires_grad=True).repeat(1,6,1,1)
        test_output = torch.mean(test_batch.clone(), dim=1)
        patched_input = extract_patches(test_batch)
        with torch.no_grad():
            example_output = example_model(patched_input)
        reconstructed = reconstruct_from_patches(example_output, 32)
        assert torch.allclose(reconstructed, test_output)