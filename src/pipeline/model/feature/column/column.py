from sklearn.pipeline import Pipeline
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline


class ColumnPipeline():
    """
    ColumnPipeline

    :param columns: list of columns
    """

    def __init__(self, columns: list[ColumnBlockPipeline]) -> None:
        """Initialize the ColumnPipeline

        :param columns: list of columns
        """
        self.columns = columns
        self.path = ""

    def get_pipeline(self) -> Pipeline:
        """Get the column pipeline

        :return: Pipeline object
        """
        steps = []

        for column in self.columns:
            if self.path:
                column.set_path(self.path)
            steps.append((str(column), column.get_pipeline()))

        if steps:
            return Pipeline(steps)

    def __str__(self) -> str:
        """String representation of the ColumnPipeline

        :return: String representation of the ColumnPipeline
        """
        return "ColumnPipeline"

    def set_path(self, path: str) -> None:
        """Set the path

        :param path: path
        """
        self.path = path
