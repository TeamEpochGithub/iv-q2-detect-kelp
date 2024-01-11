"""ModelPipeline is the class used to create the model pipeline."""

import sys
from dataclasses import dataclass

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing_extensions import Self
else:
    from typing import Self

import dask
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.pipeline.model.feature.feature import FeaturePipeline
from src.pipeline.model.model_loop.model_loop import ModelLoopPipeline
from src.pipeline.model.target.target import TargetPipeline
from src.utils.flatten_dict import flatten_dict


@dataclass
class ModelPipeline(Pipeline):
    """ModelPipeline is the class used to create the model pipeline.

    :param feature_pipeline: The feature pipeline
    :param target_pipeline: The target pipeline
    :param model_loop_pipeline: The model loop pipeline
    :param post_processing_pipeline: The post processing pipeline
    """

    feature_pipeline: FeaturePipeline | None = None
    target_pipeline: TargetPipeline | None = None
    model_loop_pipeline: ModelLoopPipeline | None = None
    post_processing_pipeline: Pipeline | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, Pipeline]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        steps = []

        if self.feature_pipeline:
            steps.append(("feature_pipeline_step", self.feature_pipeline))
        if self.target_pipeline:
            steps.append(("target_pipeline_step", self.target_pipeline))
        if self.model_loop_pipeline:
            steps.append(("model_loop_pipeline_step", self.model_loop_pipeline))
        if self.post_processing_pipeline:
            steps.append(("post_processing_pipeline_step", self.post_processing_pipeline))

        return steps

    def load_model(self, model_hashes: list[str]) -> None:
        """Load the models from the model hashes.

        :param model_hashes: The model hashes
        """
        model_hash = model_hashes[0]
        if self.model_loop_pipeline:
            self.model_loop_pipeline.load_model(model_hash)

    def load_scaler(self, scaler_hashes: list[str]) -> None:
        """Load the scalers from the scaler hashes.

        :param scaler_hashes: The scaler hashes
        """
        scaler_hash = scaler_hashes[0]
        if self.model_loop_pipeline and scaler_hash:
            self.model_loop_pipeline.load_scaler(scaler_hash)

    def save_model(self, model_hashes: list[str]) -> None:
        """Save the model to the model hash.

        :param model_hash: The model hash
        """
        model_hash = model_hashes[0]
        if self.model_loop_pipeline and model_hash:
            self.model_loop_pipeline.save_model(model_hash)

    def save_scaler(self, scaler_hashes: list[str]) -> None:
        """Save the scaler to the scaler hash.

        :param scaler_hash: The scaler hash
        """
        scaler_hash = scaler_hashes[0]
        if self.model_loop_pipeline and scaler_hash is not None:
            self.model_loop_pipeline.save_scaler(scaler_hash)

    def fit(
        self,
        X: dask.array.Array,
        y: dask.array.Array,
        train_indices: list[int] | None = None,
        test_indices: list[int] | None = None,
        cache_size: int = -1,
        model_hashes: list[str] | None = None,
    ) -> Self:
        """Fit the model pipeline.

        :param X: The input data
        :param y: The target data
        :param train_indices: The train indices
        :param test_indices: The test indices
        :param cache_size: The cache size
        :param model_hashes: The model hashes
        """
        new_params = {}

        if model_hashes and not model_hashes[0]:
            logger.info("Model hashes found. Skipping model fit.")
            return self

        if self.model_loop_pipeline:
            new_params = {
                "model_loop_pipeline_step": {
                    "model_blocks_pipeline_step": {
                        name: {"train_indices": train_indices, "test_indices": test_indices, "cache_size": cache_size}
                        for name, _ in self.model_loop_pipeline.named_steps.model_blocks_pipeline_step.steps
                    },
                }
            }

        # Add pretrain indices if it exists. Stupid mypy doesn't understand hasattr
        if self.model_loop_pipeline and hasattr(self.model_loop_pipeline.named_steps, "pretrain_pipeline_step"):
            new_params["model_loop_pipeline_step"]["pretrain_pipeline_step"] = {"train_indices": train_indices}  # type: ignore[dict-item]

        flattened = flatten_dict(new_params)

        return super().fit(X, y, **flattened)
