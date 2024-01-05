"""PostProcessingPipeline class."""
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class PostProcessingPipeline(Pipeline):
    """PostProcessingPipeline is the class used to create the post processing pipeline."""

    def __init__(self) -> None:
        """Initialize the class."""
        # TODO(Epoch): Create post processing pipeline

    def get_pipeline(self) -> Pipeline | None:
        """Get_pipeline returns the post processing pipeline.

        :return: Pipeline object
        """
        steps: list[tuple[str, BaseEstimator | Pipeline]] = []

        # TODO(Epoch): Add steps to pipeline

        if steps:
            return Pipeline(steps=steps)

        return None

    def __str__(self) -> str:
        """Overridden __str__ returns string representation of the class.

        :return: String representation of the class
        """
        return "PostProcessingPipeline"
