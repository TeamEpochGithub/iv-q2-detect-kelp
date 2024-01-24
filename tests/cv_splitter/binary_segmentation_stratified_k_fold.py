from unittest import TestCase

import numpy as np

from src.cv_splitter.binary_segmentation_stratified_k_fold import BinarySegmentationStratifiedKFold


class BinarySegmentationStratifiedKFoldTest(TestCase):
    def test_splitter(self):
        img0 = np.array([[0, 0], [0, 0]])
        img1 = np.array([[1, 0], [1, 0]])
        img2 = np.array([[1, 1], [0, 1]])
        img3 = np.array([[1, 1], [1, 1]])
        img4 = np.array([[0, 1], [0, 0]])
        img5 = np.array([[1, 0], [0, 0]])
        img6 = np.array([[0, 1], [1, 0]])
        img7 = np.array([[1, 0], [1, 1]])
        img8 = np.array([[1, 0], [0, 1]])
        img9 = np.array([[0, 0], [1, 0]])

        metadata_coverage = np.array([0, 0.5, 0.75, 1, 0.25, 0.25, 0.5, 0.75, 0.5, 0.25])

        X = np.array([img0, img1, img2, img3, img4, img5, img6, img7, img8, img9])
        y = np.copy(X)

        splitter = BinarySegmentationStratifiedKFold(n_splits=3, shuffle=False)

        folds = list(splitter.split(X, y))
        self.assertEqual(len(folds), 3)

        avg_coverage_fold0 = np.mean(y[folds[0][0]])
        avg_coverage_fold1 = np.mean(y[folds[1][0]])
        avg_coverage_fold2 = np.mean(y[folds[2][0]])

        self.assertAlmostEqual(avg_coverage_fold0, 0.5, delta=0.1)
        self.assertAlmostEqual(avg_coverage_fold1, 0.5, delta=0.1)
        self.assertAlmostEqual(avg_coverage_fold2, 0.5, delta=0.1)
