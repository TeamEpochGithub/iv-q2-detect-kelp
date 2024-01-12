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

        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        return list(self.models.items())

    def fit(
        self,
        X: da.Array,
        y: da.Array,
        train_indices: list[int] | None = None,
        test_indices: list[int] | None = None,
        cache_size: int = -1,
        *,
        save: bool = True,
    ) -> Self:
        """Fit the model.

        :param X: The input data
        :param y: The target data
        :param train_indices: The train indices
        :param test_indices: The test indices
        :param cache_size: The cache size
        :param save: Whether to save the model or not
        """
        for model in self.models.values():
            model.fit(X, y, train_indices=train_indices, test_indices=test_indices, cache_size=cache_size, save=save)
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

    def save_model(self, model_hashes: list[str]) -> None:
        """Save the model to the model hash.

        :param model_hash: The model hash
        """
        for i, model in enumerate(self.models.values()):
            model.save_model([model_hashes[i]])

    def save_scaler(self, scaler_hashes: list[str]) -> None:
        """Save the scaler to the scaler hash.

        :param scaler_hash: The scaler hash
        """
        for i, model in enumerate(self.models.values()):
            model.save_scaler([scaler_hashes[i]])
