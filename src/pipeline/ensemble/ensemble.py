"""EnsemblePipeline is the class used to create the ensemble pipeline."""
from dataclasses import dataclass, field
from typing import Any, Self

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

        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        steps = []

        # Loop through models and add them to the pipeline
        for name, model in self.models.items():
            steps.append((name, model))

        return steps

    def fit(self, X: da.Array, y: da.Array) -> Self:
        """Fit the model.

        :param X: The input data
        :param y: The target data
        """
        for model in self.models.values():
            model.fit(X, y)
        return self

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

    def load_model(self, model_hashes: list[str]) -> None:
        """Load the models from the model hashes.

        :param model_hashes: The model hashes
        """
        for i, model in enumerate(self.models.values()):
            model.load_model([model_hashes[i]])

    def load_scaler(self, scaler_hashes: list[str]) -> None:
        """Load the scalers from the scaler hashes.

        :param scaler_hashes: The scaler hashes
        """
        for i, model in enumerate(self.models.values()):
            model.load_scaler([scaler_hashes[i]])
