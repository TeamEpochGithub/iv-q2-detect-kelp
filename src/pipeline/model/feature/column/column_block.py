"""Column block pipeline."""
from dataclasses import dataclass

from joblib import hash
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.pipeline.caching.column import CacheColumnBlock


@dataclass
class ColumnBlockPipeline(Pipeline):
    """ColumnBlockPipeline extends the sklearn Pipeline class.

    :param column_block: column block
    :param cache_block: cache block
    """

    column_block: BaseEstimator
    cache_block: CacheColumnBlock | None = None

    def __post_init__(self) -> None:
        """Post init function."""
        self.path = ""
        super().__init__(self._get_steps())
        self.set_hash("")

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """Get the column block pipeline steps.

        :return: list of steps
        """
        steps = []
        if self.column_block:
            steps.append((str(self.column_block), self.column_block))
        if self.cache_block and self.path:
            self.cache_block.set_path(self.path + "/" + str(self.column_block))
            steps.append((str(self.cache_block), self.cache_block))
        return steps

    def set_path(self, path: str) -> None:
        """Set the path.

        :param path: path
        """
        self.path = path
        # Update the steps in the pipeline after changing the path
        self.steps = self._get_steps()

    def set_hash(self, prev_hash: str = "") -> str:
        """set_hash function sets the hash for the pipeline.

        :param prev_hash: previous hash
        :return: hash
        """
        column_block_hash = hash(str(self.column_block) + prev_hash)

        self.prev_hash = column_block_hash

        return column_block_hash
