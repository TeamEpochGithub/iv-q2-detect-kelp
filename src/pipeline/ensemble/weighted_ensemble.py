"""EnsemblePipeline is the class used to create the ensemble pipeline."""
import copy
from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np

from src.logging_utils.logger import logger
from src.pipeline.ensemble.ensemble_base import EnsembleBase
from src.pipeline.ensemble.error import EnsemblePipelineError


@dataclass
class WeightedEnsemble(EnsembleBase):
    """EnsemblePipeline is the class used to create the ensemble pipeline."""

    weights: list[float] = field(default_factory=list)

    def ensemble_init(self) -> None:
        """Post init function to check if the number of models and weights are the same."""
        # Check if the number of models and weights are the same
        if len(self.models) != len(self.weights):
            raise EnsemblePipelineError("The number of models and weights must be the same")

        # Normalize the weights
        self.weights = np.array(self.weights) / np.sum(self.weights)

    def transform(self, X: da.Array) -> np.ndarray[Any, Any]:
        """Transform the input data and return averaged predictions.

        :param X: The input data
        :return: The predicted target
        """
        predictions = None
        for i, model in enumerate(self.models.values()):
            if predictions is None:
                predictions = model.transform(X) * self.weights[i]
            else:
                predictions = predictions + model.transform(X) * self.weights[i]
        for step in self.post_ensemble_steps:
            predictions = step.transform(predictions)
        return np.array(predictions)

    def fit_transform(self, X: da.Array, y: da.Array, **fit_params: str) -> np.ndarray[Any, Any]:
        """Fit the pipeline and return averaged predictions.

        :param X: The input data
        :param y: The target data
        :param fit_params: The fit parameters
        :return: The averaged predictions
        """
        predictions = None
        for i, (name, model) in enumerate(self.models.items()):
            # Get the model fit params
            model_fit_params = self._get_model_fit_params(name, **fit_params)

            target_pipeline = model.get_target_pipeline()
            new_y = copy.deepcopy(y)

            if target_pipeline is not None:
                logger.info("Now fitting the target pipeline...")
                new_y = target_pipeline.fit_transform(new_y)

            curr_pred = model.fit_transform(X, new_y, **model_fit_params)

            # Check if curr_predictions is np.bool type
            if curr_pred.dtype == np.bool_:
                logger.warning("The predictions of the model are already thresholded. This will turn into majority vote...")
            if predictions is None:
                predictions = curr_pred * self.weights[i]
            else:
                predictions += curr_pred * self.weights[i]
        for step in self.post_ensemble_steps:
            # Apply the post ensemble steps
            predictions = step.fit_transform(predictions, y)

        return np.array(predictions)
