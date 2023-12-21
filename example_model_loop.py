from distributed import Client
from sklearn.pipeline import Pipeline
import torch
from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.model_loop.model_blocks.model_blocks import ModelBlocksPipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
import numpy as np
from src.utils.flatten_dict import flatten_dict
from dask_image.imread import imread
from src.pipeline.model.model_loop.model_blocks.model_fit_block import ModelBlock
import torch.nn as nn
from dask import config as cfg
import coloredlogs

coloredlogs.install()
cfg.set({'distributed.scheduler.worker-ttl': None})


if __name__ == "__main__":
    # This is meant to be an example of how to set up the model loop pipeline
    # Do not remove this code

    # make a nn.Module model
    Client(n_workers=24, threads_per_worker=1, memory_limit='16GB')

    model = nn.Conv2d(9, 1, 3, padding=1)
    # make a optimizer
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # make a scheduler
    scheduler = None
    # make a loss function

    class DiceLoss(nn.Module):
        def __init__(self, size_average: bool = True) -> None:
            super().__init__()

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> float:

            # comment out if your model contains a sigmoid or equivalent activation layer
            # inputs = F.sigmoid(inputs)

            # flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            intersection = (inputs * targets).sum()
            dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

            return 1 - dice
    criterion = DiceLoss()

    # make a model fit block

    model_fit_block = ModelBlock(model, optimizer, scheduler, criterion, epochs=1, batch_size=32, patience=10)
    model_str = str(model_fit_block)
    # make a model blocks pipeline
    model_blocks_pipeline = ModelBlocksPipeline(model_blocks=[model_fit_block])

    model_loop_pipeline = ModelLoopPipeline(None, model_blocks_pipeline=model_blocks_pipeline)

    raw_data_path = "data/raw/train_satellite"
    processed_path = "data/processed"
    features_path = "data/features"
    transform_steps = [{'type': 'divider', 'divider': 65500}]
    columns = [{'type': 'band_copy', 'band': 0},
               {'type': 'band_copy', 'band': 2}]
    feature_pipeline = FeaturePipeline(
        raw_data_path, processed_path, transformation_steps=transform_steps, column_steps=columns)
    fp = feature_pipeline.get_pipeline()
    mp = model_loop_pipeline.get_pipeline()
    pipeline = Pipeline(steps=[('feature_pipeline', fp), ('model_loop_pipeline', mp)])

    indices = np.arange(5635)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:4508], indices[4508:]
    fit_params = {
        "model_loop_pipeline": {
            "model_blocks_pipeline": {
                model_str: {
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "to_mem_length": 3000
                },

            }
        }
    }

    predict_params = {
        "to_mem_length": 0
    }
    y = imread("data/raw/train_kelp/*.tif")
    pipeline.fit(None, y, **flatten_dict(fit_params))
    pipeline.predict(None, **flatten_dict(predict_params))
