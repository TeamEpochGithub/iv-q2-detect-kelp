from distributed import Client
from sklearn.pipeline import Pipeline
import torch
from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.model_loop.model_blocks.model_blocks import ModelBlocksPipeline
from src.pipeline.model.model_loop.pretrain.pretrain import PretrainPipeline
import numpy as np
from src.utils.flatten_dict import flatten_dict
import torch.nn.functional as F


class ModelLoopPipeline():
    """
    Model loop pipeline.
    :param pretrain_pipeline: Pretrain pipeline.
    :param model_blocks_pipeline: Model blocks pipeline.
    """
    def __init__(self, pretrain_pipeline: PretrainPipeline | None, model_blocks_pipeline: ModelBlocksPipeline | None):
        """
        Model loop pipeline.
        :param pretrain_pipeline: Pretrain pipeline.
        :param model_blocks_pipeline: Model blocks pipeline.
        """
        self.pretrain_pipeline = pretrain_pipeline
        self.model_blocks_pipeline = model_blocks_pipeline

    def get_pipeline(self, cache_model: bool = True) -> Pipeline:
        """
        Get the pipeline.
        :param cache_model: Whether to cache the model.
        :return: Pipeline object."""
        steps = []

        if self.pretrain_pipeline:
            steps.append(('pretrain_pipeline', self.pretrain_pipeline.get_pipeline()))
        if self.model_blocks_pipeline:
            steps.append(('model_blocks_pipeline', self.model_blocks_pipeline.get_pipeline()))

        return Pipeline(steps=steps, memory="tm" if cache_model else None)


if __name__ == "__main__":
    # This is meant to be an example of how to set up the model loop pipeline
    # Do not remove this code
    
    # make a nn.Module model
    Client()
    import torch.nn as nn
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
            inputs = F.sigmoid(inputs)

            # flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            intersection = (inputs * targets).sum()
            dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

            return 1 - dice
    criterion = DiceLoss()

    # make a model fit block
    from src.pipeline.model.model_loop.model_blocks.model_fit_block import ModelFitBlock
    model_fit_block = ModelFitBlock(model, optimizer, scheduler, criterion, epochs=1, batch_size=32, patience=10)
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
    print(pipeline)

    indices = np.arange(5635)
    train_indices, test_indices = indices[:4508], indices[4508:]
    fit_params = {
        "model_loop_pipeline": {
            "model_blocks_pipeline": {
                model_str: {
                    "train_indices": train_indices,
                    "test_indices": test_indices,
                    "to_mem_length": 0
                },

            }
        }
    }

    predict_params = {
        "to_mem_length": 5635
    }

    pipeline.predict(None, **flatten_dict(predict_params))

    # this will crash because the label pipeline isnt done yet
    pipeline.fit(None, None, **flatten_dict(fit_params))
