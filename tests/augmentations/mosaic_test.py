from unittest import TestCase

import numpy as np

from src.augmentations.mosaic import Mosaic


class Test(TestCase):

    def test_mosaic(self):
        # Test the mosaic augmentation with 4 random images

        # Create 4 random images and masks where each image has 1 color

        image_1 = np.random.randint(0, 50, (256, 256, 3))
        image_2 = np.random.randint(50, 150, (256, 256, 3))
        image_3 = np.random.randint(150, 200, (256, 256, 3))
        image_4 = np.random.randint(200, 255, (256, 256, 3))

        mask_1 = np.random.randint(0, 50, (256, 256, 1))
        mask_2 = np.random.randint(50, 150, (256, 256, 1))
        mask_3 = np.random.randint(150, 200, (256, 256, 1))
        mask_4 = np.random.randint(200, 255, (256, 256, 1))

        # Create the images and masks arrays
        images = np.array([image_1, image_2, image_3, image_4])
        masks = np.array([mask_1, mask_2, mask_3, mask_4])

        # Create the mosaic augmentation
        mosaic = Mosaic(p=1)

        # Apply the mosaic augmentation
        image, mask = mosaic.mosaic(images, masks, [0, 1, 2, 3])

        true_shape_image = (256, 256, 3)
        true_shape_mask = (256, 256, 1)

        self.assertEqual(image.shape, true_shape_image)
        self.assertEqual(mask.shape, true_shape_mask)

        # Assert that all values are in image

        # Check if < 49 exists in image
        self.assertTrue(np.any(image < 49))
        # Check if > 50 and i < 150 exists in image
        self.assertTrue(np.any(np.logical_and(image > 50, image < 150)))

        # Check if > 150 and i < 200 exists in image
        self.assertTrue(np.any(np.logical_and(image > 150, image < 200)))

        # Check if > 200 and i < 255 exists in image
        self.assertTrue(np.any(np.logical_and(image > 200, image < 255)))

