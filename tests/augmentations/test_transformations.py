from unittest import TestCase

import numpy as np

from src.augmentations.mosaic import Mosaic
from src.augmentations.transformations import Transformations


class Test(TestCase):

    def color_image_gen(self, n, c):
        #Generate n images with unique colors
        img_shape = (c, 256, 256)
        if c == 1:
            img_shape = (256, 256)
        all_images = []
        r = 256 // n
        for i in range(n):
            img = np.random.randint(i*r, i*r+r, img_shape)
            all_images.append(img)
        return np.array(all_images)



    def test_transformations_16batch(self):


        mosaic = Mosaic(p=1)
        trans = Transformations(alb=None, aug=[mosaic])

        images = self.color_image_gen(16, 3)
        masks = self.color_image_gen(16, 1)
        print(images.shape)
        print(masks.shape)

        transformed = trans.transform(images, masks)
        for i, (image, mask) in enumerate(zip(*transformed)):

            import cv2
            #Save to file
            # cv2.imwrite(f"test_{i}_image.png", image.transpose(1, 2, 0))
            # cv2.imwrite(f"test_{i}_mask.png", mask)

            self.assertEqual(image.shape, (3, 256, 256))
            self.assertEqual(mask.shape, (256, 256))

    def test_transformations_4batch(self):

        mosaic = Mosaic(p=1)
        trans = Transformations(alb=None, aug=[mosaic])


        images = self.color_image_gen(4, 3)
        masks = self.color_image_gen(4, 1)

        transformed = trans.transform(images, masks)

        print(transformed[0].shape)
        print(transformed[1].shape)
        for i, (image, mask) in enumerate(zip(*transformed)):

            import cv2
            #Save to file
            # cv2.imwrite(f"test_{i}_image.png", image.transpose(1, 2, 0))
            # cv2.imwrite(f"test_{i}_mask.png", mask)

            self.assertEqual(image.shape, (3,256,256))
            self.assertEqual(mask.shape,(256,256))

            #Assert that all values are in image

            #Check if < 49 exists in image
            self.assertTrue(np.any(image < 49))
            # Check if > 50 and i < 150 exists in image
            self.assertTrue(np.any(np.logical_and(image > 50, image < 150)))

            # Check if > 150 and i < 200 exists in image
            self.assertTrue(np.any(np.logical_and(image > 150, image < 200)))

            # Check if > 200 and i < 255 exists in image
            self.assertTrue(np.any(np.logical_and(image > 200, image < 255)))



