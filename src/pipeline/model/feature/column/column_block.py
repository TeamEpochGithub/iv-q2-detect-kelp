from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from src.pipeline.caching.column import CacheColumnBlock


class ColumnBlockPipeline(Pipeline):
    """
    ColumnBlockPipeline

    :param column_block: column block
    :param cache_block: cache block
    """

    def __init__(self, column_block: BaseEstimator, cache_block: CacheColumnBlock | None = None) -> None:
        """
        Initialize the ColumnBlockPipeline

        :param column_block: column block
        :param cache_block: cache block
        """
        self.column_block = column_block
        self.cache_block = cache_block
        self.path = ""
        super().__init__(self._get_steps())

    def _get_steps(self) -> list[tuple[str, BaseEstimator | Pipeline]]:
        """
        Get the column block pipeline steps

        :return: list of steps
        """
        steps = []
        if self.column_block:
            steps.append((str(self.column_block), self.column_block))
        if self.cache_block:
            if self.path:
                self.cache_block.set_path(self.path + "/" + str(self.column_block))
                steps.append((str(self.cache_block), self.cache_block))
        return steps

    def set_path(self, path: str) -> None:
        """Set the path

        :param path: path
        """
        self.path = path
        # Update the steps in the pipeline after changing the path
        self.steps = self._get_steps()

    def __str__(self) -> str:
        """Convert the class to a string

        :return: string representation of the class
        """
        return f"ColumnBlockPipeline({self.column_block}, {self.cache_block})"
