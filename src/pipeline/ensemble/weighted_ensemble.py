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
            model_fit_params = {key: value for key, value in fit_params.items() if key.startswith(name)}
            # Remove the model name from the fit params key
            model_fit_params = {key[len(name) + 2 :]: value for key, value in model_fit_params.items()}

            target_pipeline = model.get_target_pipeline()
            new_y = copy.deepcopy(y)

            if target_pipeline is not None:
                logger.info("Now fitting the target pipeline...")
                new_y = target_pipeline.fit_transform(new_y)

            if predictions is None:
                predictions = model.fit_transform(X, new_y, **model_fit_params) * self.weights[i]
            else:
                predictions = predictions + model.fit_transform(X, new_y, **model_fit_params) * self.weights[i]
        for step in self.post_ensemble_steps:
            predictions = step.transform(predictions)

        # Make sure predictions are 0 or 1, use threshold of 0.5
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        return np.array(predictions)
