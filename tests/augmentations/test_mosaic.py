from unittest import TestCase

import numpy as np

from src.augmentations.mosaic import Mosaic


class Test(TestCase):

    def test_mosaic(self):
        # Test the mosaic augmentation with 4 random images

        # Create 4 random images and masks where each image has 1 color


        img_shape = (3, 256, 256)
        mask_shape = (1, 256, 256)
        image_1 = np.random.randint(0, 50, img_shape)
        image_2 = np.random.randint(50, 150, img_shape)
        image_3 = np.random.randint(150, 200, img_shape)
        image_4 = np.random.randint(200, 255, img_shape)

        mask_1 = np.random.randint(0, 50, mask_shape)
        mask_2 = np.random.randint(50, 150, mask_shape)
        mask_3 = np.random.randint(150, 200, mask_shape)
        mask_4 = np.random.randint(200, 255, mask_shape)

        # Create the images and masks arrays
        images = np.array([image_1, image_2, image_3, image_4])
        masks = np.array([mask_1, mask_2, mask_3, mask_4])

        # Create the mosaic augmentation
        mosaic = Mosaic(p=1)

        # Apply the mosaic augmentation
        image, mask = mosaic.mosaic(images, masks, [0, 1, 2, 3])


        self.assertEqual(image.shape, img_shape)
        self.assertEqual(mask.shape, mask_shape)

        # Assert that all values are in image

        # Check if < 49 exists in image
        self.assertTrue(np.any(image < 49))
        # Check if > 50 and i < 150 exists in image
        self.assertTrue(np.any(np.logical_and(image > 50, image < 150)))

        # Check if > 150 and i < 200 exists in image
        self.assertTrue(np.any(np.logical_and(image > 150, image < 200)))

        # Check if > 200 and i < 255 exists in image
        self.assertTrue(np.any(np.logical_and(image > 200, image < 255)))


        # save  the images with cv
        import cv2


