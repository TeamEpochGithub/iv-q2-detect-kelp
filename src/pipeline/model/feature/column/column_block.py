from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from src.pipeline.caching.column import CacheColumnBlock


class ColumnBlockPipeline():
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
        self.path = None

    def get_pipeline(self) -> Pipeline:
        """
        Get the column block pipeline

        :return: Pipeline object
        """
        steps = []

        if self.column_block:
            steps.append((str(self.column_block), self.column_block))
        if self.cache_block:
            if self.path:
                self.cache_block.set_path(self.path + "/" + str(self.column_block))
            steps.append((str(self.cache_block), self.cache_block))
            memory = self.cache_block.get_data_path() + "/pipeline"

        if steps:
            return Pipeline(steps, memory=memory)

    def __str__(self) -> str:
        """
        String representation of the ColumnBlockPipeline

        :return: String representation of the ColumnBlockPipeline
        """
        return f"ColumnBlockPipeline_{str(self.column_block)}"

    def set_path(self, path: str) -> None:
        """
        Set the path

        :param path: path
        """
        self.path = path

if __name__ == "__main__":
    
    from src.pipeline.model.feature.column.band_copy import BandCopy
    band_copy_pipeline = BandCopy(1)

    from src.pipeline.caching.column import CacheColumnBlock
    cache = CacheColumnBlock(
                "data/test", column=-1)
    column_block_pipeline = ColumnBlockPipeline(band_copy_pipeline, cache)
    import numpy as np
    X = np.array([[1,2], [3,4]])
    X = column_block_pipeline.get_pipeline().fit_transform(X)
    print(X.compute())