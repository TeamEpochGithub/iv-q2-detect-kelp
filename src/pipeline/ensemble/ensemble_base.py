"""Base class for ensemble pipelines."""
import copy
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.pipeline.model.model import ModelPipeline

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self


@dataclass
class EnsembleBase(Pipeline):
    """Base class for ensemble pipelines.

    :param models: The models to use in the ensemble
    :param post_ensemble_steps: The steps to apply after the ensemble
    """

    models: dict[str, ModelPipeline] = field(default_factory=dict)
    post_ensemble_steps: list[Pipeline | BaseEstimator] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function."""
        self.ensemble_init()
        if self.post_ensemble_steps is None:
            self.post_ensemble_steps = []
        super().__init__(self._get_steps())

    @abstractmethod
    def ensemble_init(self) -> None:
        """Ensemble init function."""

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        steps = list(self.models.items())
        if self.post_ensemble_steps:
            steps.extend([("post_ensemble_step", step) for step in self.post_ensemble_steps])
        return steps

    def predict(self, X: da.Array) -> np.ndarray[Any, Any]:
        """Predict the target for each model and average the predictions by weight.

        :param X: The input data
        :return: The predicted target
        """
        return self.transform(X)

    @abstractmethod
    def transform(self, X: da.Array) -> np.ndarray[Any, Any]:
        """Transform the input data and return averaged predictions.

        :param X: The input data
        :return: The transformed data
        """

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

            target_pipeline = model.get_target_pipeline()
            new_y = copy.deepcopy(y)

            if target_pipeline is not None:
                logger.info("Now fitting the target pipeline...")
                new_y = target_pipeline.fit_transform(new_y)

            model.fit(X, new_y, **model_fit_params)
        return self

    @abstractmethod
    def fit_transform(self, X: da.Array, y: da.Array, **fit_params: str) -> np.ndarray[Any, Any]:
        """Fit the pipeline and return averaged predictions.

        :param X: The input data
        :param y: The target data
        :param fit_params: The fit parameters
        :return: The averaged predictions
        """

    def _get_model_fit_params(self, name: str, **fit_params: str) -> dict[str, Any]:
        """Get the model fit params.

        :param name: The name of the model
        :param fit_params: The fit parameters
        :return: The model fit parameters in the correct format
        """
        model_fit_params = {key: value for key, value in fit_params.items() if key.startswith(name)}
        # Remove the model name from the fit params key
        return {key[len(name) + 2 :]: value for key, value in model_fit_params.items()}
