"""Deep learning ensemble pipeline step.

This module contains the deep learning ensemble pipeline step.
Will only work on model pipelines that have the model as the last step.
This is because predict is called using the feature map argument, which is only available in the model pipeline.
"""
import copy
import functools
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import dask.array as da
import numpy as np
from annotated_types import Gt, Interval
from joblib import hash
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.augmentations.transformations import Transformations
from src.logging_utils.logger import logger
from src.pipeline.ensemble.ensemble_base import EnsembleBase
from src.pipeline.model.model_loop.model_blocks.auxiliary_block import AuxiliaryBlock
from src.modules.loss.auxiliary_loss_double import AuxiliaryLossDouble
from functools import partial

from src.pipeline.model.model_loop.model_blocks.torch_block import TorchBlock


@dataclass
class DLEnsemble(EnsembleBase):
    optimizer: functools.partial[Optimizer] = field(default_factory=partial(Optimizer, lr=0.001))
    scheduler: Callable[[Optimizer], LRScheduler] | None = None
    criterion: nn.Module = field(default_factory=nn.Module())
    epochs: Annotated[int, Gt(0)] = 10
    batch_size: Annotated[int, Gt(0)] = 32
    patience: Annotated[int, Gt(0)] = 5
    # noinspection PyTypeHints
    test_size: Annotated[float, Interval(ge=0, le=1)] = 0.2  # Hashing purposes
    transformations: Transformations | None = None
    layerwise_lr_decay: float | None = None

    def ensemble_init(self) -> None:
        self.ensemble_hash = hash(self)
        self.feature_map_args = {
            "feature_map": True,
        }

    def transform(self, X: da.Array) -> np.ndarray[Any, Any]:
        """Transform the input data and return averaged predictions.

        :param X: The input data
        :return: The predicted target
        """
        model_predictions = []
        for i, model in enumerate(self.models.values()):
            model_predictions.extend(model.predict(X, **self.feature_map_args))

        # Pass through the segmentation head
        if self.segmentation_head is None:
            raise ValueError("Segmentation head not trained yet")
        segmented_predictions = self.segmentation_head(model_predictions)

        # Union
        regression_preds = segmented_predictions[:, 0] > 0.5
        classification_preds = segmented_predictions[:, 1:].argmax(axis=1)

        stacked_preds = np.stack([regression_preds, classification_preds], axis=1)

        # Perform the logical OR operation on the two channels
        union_preds = np.logical_or(stacked_preds[:, 0], stacked_preds[:, 1])

        # Convert the boolean array to an integer array
        union_preds = union_preds.astype(np.uint8)

        predictions = union_preds
        for step in self.post_ensemble_steps:
            predictions = step.transform(union_preds)
        return np.array(predictions)

    def fit_transform(self, X: da.Array, y: da.Array, **fit_params: str) -> np.ndarray[Any, Any]:
        """Fit the pipeline and return averaged predictions.

        :param X: The input data
        :param y: The target data
        :param fit_params: The fit parameters
        :return: The averaged predictions
        """
        # Train the models if needed and get the feature maps
        for i, (name, model) in enumerate(self.models.items()):
            model_fit_params = {key: value for key, value in fit_params.items() if key.startswith(name)}
            # Remove the model name from the fit params key
            model_fit_params = {key[len(name) + 2 :]: value for key, value in model_fit_params.items()}

            target_pipeline = model.get_target_pipeline()
            new_y = copy.deepcopy(y)

            if target_pipeline is not None:
                logger.info("Now fitting the target pipeline...")
                new_y = target_pipeline.fit_transform(new_y)

            model.fit(X, new_y, **model_fit_params)

        # Create empty numpy array
        model_predictions = np.empty((X.shape[0], 0,350,350))

        for model in self.models.values():
            model_predictions = np.concatenate([model_predictions, model.predict(X, **self.feature_map_args)], axis=1)

        # Train the segmentation head if needed
        self.segmentation_head = self._create_segmentation_head(model_predictions.shape)

        # Get fit arguments
        train_indices = fit_params["train_indices"] if "train_indices" in fit_params else None
        test_indices = fit_params["test_indices"] if "test_indices" in fit_params else None

        predictions = self.segmentation_head.fit_transform(model_predictions, y, train_indices=train_indices, test_indices=test_indices)

        for step in self.post_ensemble_steps:
            predictions = step.transform(predictions)

        return np.array(predictions)

    def _create_segmentation_head(self, input_shape: tuple[int, int, int, int]) -> nn.Module:
        """Create the segmentation head.

        :param input_shape: The input shape
        :return: The segmentation head
        """
        segmentation_head = nn.Sequential(
            nn.Conv2d(input_shape[1], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid(),
        )

        # Create auxiliary block
        aux_block = TorchBlock(
            model=segmentation_head,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=self.criterion,
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            test_size=self.test_size,
            transformations=self.transformations,
            layerwise_lr_decay=self.layerwise_lr_decay,
        )

        aux_block.set_hash(self.ensemble_hash)

        return aux_block
