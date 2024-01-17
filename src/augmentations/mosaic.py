"""Implementation of the Mosaic augmentation."""

from dataclasses import dataclass

import numpy as np
from pandas._typing import npt

from src.augmentations.augmentation import Augmentation


@dataclass
class Mosaic(Augmentation):
    """Implementation of the Mosaic augmentation."""

    p: float
    img_to_apply: int = 4

    def transforms(self, images: npt.NDArray[np.float_], masks: npt.NDArray[np.float_], i: int) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the augmentation to the data.

        :param x: Batch of input features.
        :param y: Batch of labels.
        :param i: Index of the image to apply the augmentation to.
        :return: Augmentation applied to the image and mask at index i
        """
        # Get a random float between 0 and 1
        r = np.random.rand()
        # If the random float is less than the probability of the augmentation, apply it

        if r < self.p:
            # Get the indices of the images to apply the augmentation to
            idxs = np.random.randint(0, len(images), self.img_to_apply)
            # Apply the mosaic augmentation
            image, mask = self.mosaic(images, masks, idxs)
            # Return the augmented image and mask
            return image, mask

        else:
            # If the augmentation is not applied, return the original image and mask
            return images[i], masks[i]

    def mosaic(self, images: npt.NDArray[np.float_], masks: npt.NDArray[np.float_], idxs: list[int]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Apply the mosaic augmentation to the images and masks.

        :param images: Batch of input features.
        :param masks: Batch of labels.
        :param idxs: Indices of the images to apply the mosaic augmentation to.
        :return: Augmented images and masks
        """
        # Get the images and masks to apply the mosaic to
        image_1, mask_1 = images[idxs[0]], masks[idxs[0]]
        image_2, mask_2 = images[idxs[1]], masks[idxs[1]]
        image_3, mask_3 = images[idxs[2]], masks[idxs[2]]
        image_4, mask_4 = images[idxs[3]], masks[idxs[3]]

        # Get the height and width of the images
        h, w = image_1.shape[0], image_1.shape[1]

        # Create the mosaic image and mask, ensure that the mosaic image and mask have the same shape as the original images and masks
        image = np.zeros((image_1.shape[0], image_1.shape[1] * 2, image_1.shape[2] * 2))
        mask = np.zeros((mask_1.shape[0], mask_1.shape[1] * 2, mask_1.shape[2] * 2))

        # Apply the mosaic augmentation
        image[:, :h, :w] = image_1
        image[:, :h, w:] = image_2
        image[:, h:, :w] = image_3
        image[:, h:, w:] = image_4

        mask[:, :h, :w] = mask_1
        mask[:, :h, w:] = mask_2
        mask[:, h:, :w] = mask_3
        mask[:, h:, w:] = mask_4

        # Now make the cutout from the final image to get back to original image width and height
        # Since the image is twice the input size, we can get
        # a random value between root of the size and the size
        # to get a random (x, y) coordinate of the
        # top left of the cut. Then add the size
        # to get the bottom right of the cut

        # Get the random x and y coordinates
        topL_x = np.random.randint(int(np.sqrt(h)), h - int(np.sqrt(h)))
        topL_y = np.random.randint(int(np.sqrt(w)), w - int(np.sqrt(w)))
        cut_x = topL_x + h
        cut_y = topL_y + w

        # Cut the image and mask
        image = image[:, topL_x:cut_x, topL_y:cut_y]
        mask = mask[:, topL_x:cut_x, topL_y:cut_y]

        return image, mask
