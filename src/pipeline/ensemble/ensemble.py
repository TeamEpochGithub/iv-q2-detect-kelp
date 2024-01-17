"""EnsemblePipeline is the class used to create the ensemble pipeline."""
import sys
from dataclasses import dataclass, field
from typing import Any

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self

import dask.array as da
import numpy as np
from sklearn.pipeline import Pipeline

from src.pipeline.ensemble.error import EnsemblePipelineError
from src.pipeline.model.model import ModelPipeline


@dataclass
class EnsemblePipeline(Pipeline):
    """EnsemblePipeline is the class used to create the ensemble pipeline."""

    models: dict[str, ModelPipeline] = field(default_factory=dict)
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function to check if the number of models and weights are the same."""
        # Check if the number of models and weights are the same
        if len(self.models) != len(self.weights):
            raise EnsemblePipelineError("The number of models and weights must be the same")

        # Normalize the weights
        self.weights = np.array(self.weights) / np.sum(self.weights)

        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        return list(self.models.items())

    def predict(self, X: da.Array) -> np.ndarray[Any, Any]:
        """Predict the target for each model and average the predictions by weight.

        :param X: The input data
        :return: The predicted target
        """
        predictions = None
        for i, model in enumerate(self.models.values()):
            if predictions is None:
                predictions = model.transform(X) * self.weights[i]
            else:
                predictions = predictions + model.transform(X) * self.weights[i]
        return np.array(predictions)

    def transform(self, X: da.Array) -> np.ndarray[Any, Any]:
        """Transform the input data and return averaged predictions.

        :param X: The input data
        :return: The transformed data
        """
        return self.predict(X)

    def fit(self, X: da.Array, y: da.Array, **fit_params: str) -> Self:
        """Fit the pipeline.

        :param X: The input data
        :param y: The target data
        :param fit_params: The fit parameters
        :return: The fitted pipeline
        """
        for name, model in self.models.items():
            model_fit_params = {key: value for key, value in fit_params.items() if key.startswith(name)}
            # Remove the model name from the fit params key
            model_fit_params = {key[len(name) + 2 :]: value for key, value in model_fit_params.items()}
            model.fit(X, y, **model_fit_params)
        return self

    def fit_transform(self, X: da.Array, y: da.Array, **fit_params: str) -> np.ndarray[Any, Any]:
        """Fit the pipeline and return averaged predictions.

        :param X: The input data
        :param y: The target data
        :param fit_params: The fit parameters
        :return: The averaged predictions
        """
        predictions = None
        for i, model in enumerate(self.models.values()):
            if predictions is None:
                predictions = model.fit_transform(X, y, **fit_params) * self.weights[i]
            else:
                predictions = predictions + model.transform(X) * self.weights[i]
        return np.array(predictions)
