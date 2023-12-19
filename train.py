"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
from dask_image.imread import imread
import numpy as np
from sklearn.model_selection import train_test_split
from src.pipeline.model.feature.feature import FeaturePipeline

if __name__ == '__main__':
    # Temporary args before config file is implemented TODO remove
    split = 0.2
    raw_feature_path = 'data/raw/train_satellite'
    raw_target_path = 'data/raw/train_kelp'
    processed_path = 'data/processed'
    transformation_steps = []
    column_steps = []

    # Get feature pipeline
    fp = FeaturePipeline(raw_feature_path, processed_path,
                         transformation_steps=transformation_steps, column_steps=column_steps)
    feature_pipeline = fp.get_pipeline()
    x = feature_pipeline.fit_transform(None)

    # Get target pipeline TODO
    y = imread(f"{raw_target_path}/*.tif")
    # Create an array of indices
    indices = np.arange(x.shape[0])

    # Split indices into train and test
    train_indices, test_indices = train_test_split(indices, test_size=split)

    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")

    # Create a KFold object
    # kf = KFold(n_splits=5)

    # # Split indices into train and test for each fold
    # for train_indices, test_indices in kf.split(indices):
    #     print(f"Train indices: {train_indices}")
    #     print(f"Test indices: {test_indices}")


