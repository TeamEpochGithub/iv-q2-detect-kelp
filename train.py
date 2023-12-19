"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
from dask_image.imread import imread
from distributed import Client
import numpy as np
from sklearn.model_selection import train_test_split
from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.model_loop.model_fit_block import ModelFitBlock
import torch
import torch.nn as nn


if __name__ == '__main__':
    # Temporary args before config file is implemented TODO remove
    client = Client()
    # Print port
    print(client.scheduler_info()['services'])

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

    # create a torch model
    model = nn.Conv2d(7, 1, 3, padding=1)
    # create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # create a scheduler
    scheduler = None
    # create a loss function based on dice loss
    import torch.nn.functional as F
    class DiceLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(DiceLoss, self).__init__()

        def forward(self, inputs, targets, smooth=1):
            
            #comment out if your model contains a sigmoid or equivalent activation layer
            inputs = F.sigmoid(inputs)       
            
            #flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            intersection = (inputs * targets).sum()                            
            dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
            
            return 1 - dice
    criterion = DiceLoss()
    # create a model fit block
    model_fit_block = ModelFitBlock(model, optimizer, scheduler, criterion)

    # from sklearn.pipeline import Pipeline
    # model_loop_pipeline = Pipeline(steps=[('model_fit_block', model_fit_block)], memory="tm")

    # model_pipeline = Pipeline(steps=[('feature_pipeline', feature_pipeline), ('model_loop_pipeline', model_loop_pipeline)])
    # model_args = {

    # }
    # model_pipeline.fit(None, None, **model_args)
    # fit the model
    model_fit_block.fit(x, y, train_indices, test_indices, 20, 32, 10, 5635)