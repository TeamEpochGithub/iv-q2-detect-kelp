from sklearn.pipeline import Pipeline
from src.pipeline.model.feature.column.column_block import ColumnBlockPipeline


class ColumnPipeline():
    """
    ColumnPipeline

    :param columns: list of columns
    """

    def __init__(self, columns: list[ColumnBlockPipeline]) -> None:
        """
        Initialize the ColumnPipeline

        :param columns: list of columns
        """
        self.columns = columns

    def get_pipeline(self) -> Pipeline:
        """
        Get the column pipeline

        :return: Pipeline object
        """
        steps = []

        for column in self.columns:
            steps.append((str(column), column.get_pipeline()))

        if steps:
            return Pipeline(steps)

    def __str__(self) -> str:
        """
        String representation of the ColumnPipeline

        :return: String representation of the ColumnPipeline
        """
        return "ColumnPipeline"


if __name__ == "__main__":
    from src.pipeline.model.feature.column.band_copy import BandCopy
    band_copy_pipeline = BandCopy(1)

    from src.pipeline.caching.column import CacheColumnBlock
    cache = CacheColumnBlock(
        "data/test", column=-1)
    column_block_pipeline = ColumnBlockPipeline(band_copy_pipeline, cache)
    column_pipeline = ColumnPipeline([column_block_pipeline])
    print(column_pipeline.get_pipeline())

    import numpy as np
    X = np.array([[1, 2], [3, 4]])
    X = column_pipeline.get_pipeline().fit_transform(X)
    print(X.compute())
