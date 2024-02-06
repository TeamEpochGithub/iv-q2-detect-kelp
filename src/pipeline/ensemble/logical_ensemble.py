"""LogicalEnsemble is the class used to create the ensemble pipeline. Uses union/intersection of thresholded predictions instead of weighted average."""
import copy
from dataclasses import dataclass, field
from typing import Any

import dask.array as da
import numpy as np

from src.logging_utils.logger import logger
from src.pipeline.ensemble.ensemble_base import EnsembleBase
from src.pipeline.ensemble.error import EnsemblePipelineError


@dataclass
class LogicalEnsemble(EnsembleBase):
    """LogicalEnsemble is the class used to create the ensemble pipeline. Uses union/intersection of thresholded predictions instead of weighted average."""

    ensemble_type: str = field(default_factory=str)

    def ensemble_init(self) -> None:
        """Post init function to check if the type is either union or intersection."""
        if self.ensemble_type not in ["union", "intersection"]:
            raise EnsemblePipelineError("The type of logical ensemble must be either union or intersection")

    def transform(self, X: da.Array) -> np.ndarray[Any, Any]:
        """Transform the input data and return union predictions.

        :param X: The input data
        :return: The predicted target
        """
        predictions = None
        for model in self.models.values():
            if predictions is None:
                predictions = model.transform(X)
            else:
                curr_preds = model.transform(X)

                # Raise an error if the curr_preds is not a boolean numpy array
                if curr_preds.dtype != np.bool_:
                    raise EnsemblePipelineError("The predictions must be a boolean numpy array")
                if self.ensemble_type == "union":
                    predictions = np.logical_or(predictions, curr_preds)
                elif self.ensemble_type == "intersection":
                    predictions = np.logical_and(predictions, curr_preds)

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
        for name, model in self.models.items():
            # Get the model fit params
            model_fit_params = self._get_model_fit_params(name, **fit_params)

            target_pipeline = model.get_target_pipeline()
            new_y = copy.deepcopy(y)

            if target_pipeline is not None:
                logger.info("Now fitting the target pipeline...")
                new_y = target_pipeline.fit_transform(new_y)

            logger.info("")
            if predictions is None:
                predictions = model.fit_transform(X, new_y, **model_fit_params)
            else:
                curr_preds = model.fit_transform(X, new_y, **model_fit_params)

                # Raise an error if the curr_preds is not a boolean numpy array
                if curr_preds.dtype != np.bool_:
                    raise EnsemblePipelineError("The predictions must be a boolean numpy array")
                if self.ensemble_type == "union":
                    predictions = np.logical_or(predictions, curr_preds)
                elif self.ensemble_type == "intersection":
                    predictions = np.logical_and(predictions, curr_preds)

        for step in self.post_ensemble_steps:
            # Apply the post ensemble steps
            predictions = step.fit_transform(predictions, y)
        return np.array(predictions)
