"""Pretrain pipeline class."""
import time
from dataclasses import dataclass
from typing import Any

import dask.array as da
from sklearn.pipeline import Pipeline

from src.logging_utils.logger import logger
from src.logging_utils.section_separator import print_section_separator
from src.pipeline.model.model_loop.pretrain.pretrain_block import PretrainBlock


@dataclass
class PretrainPipeline(Pipeline):
    """Class used to create the pretrain pipeline.

    :param steps: list of steps
    """

    pretrain_steps: list[PretrainBlock]
    pretrain_path: str | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        self.set_hash("")

        # Set processed path
        if self.pretrain_path:
            for step in self.pretrain_steps:
                step.set_pretrain_path(self.pretrain_path)

        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, PretrainBlock]]:
        """Get the pipeline steps.

        :return: list of steps
        """
        return [(str(step), step) for step in self.pretrain_steps]

    def set_hash(self, prev_hash: str) -> str:
        """Set the hash.

        :param prev_hash: Previous hash
        :return: Hash
        """
        pretrain_hash = prev_hash
        for step in self.pretrain_steps:
            pretrain_hash = step.set_hash(pretrain_hash)

        self.prev_hash = pretrain_hash

        return pretrain_hash

    def fit_transform(self, X: da.Array, y: da.Array | None = None, **fit_params: dict[str, Any]) -> da.Array:
        """Fit and transform the data.

        :param X: Data to fit and transform
        :param y: Target data
        :param fit_params: Fit parameters
        """
        print_section_separator("Pretrain")
        start_time = time.time()
        X = super().fit_transform(X, y, **fit_params)
        logger.info(f"Fitted pretrain pipeline in {time.time() - start_time} seconds")
        return X

    def transform(self, X: da.Array) -> da.Array:
        """Transform the data.

        :param X: Data to transform
        """
        print_section_separator("Pretrain")
        start_time = time.time()
        X = super().transform(X)
        logger.info(f"Transformed pretrain pipeline in {time.time() - start_time} seconds")
        return X
