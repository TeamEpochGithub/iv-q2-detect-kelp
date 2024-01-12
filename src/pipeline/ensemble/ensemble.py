"""EnsemblePipeline is the class used to create the ensemble pipeline."""
from dataclasses import dataclass, field
from typing import Any

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
