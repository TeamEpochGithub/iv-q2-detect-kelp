"""TargetPipeline class sets up the target pipeline."""
from sklearn.pipeline import Pipeline


class TargetPipeline:
    """TargetPipeline is the class used to create the target pipeline.

    :param raw_target_path: The raw target path
    :param processed_path: The processed path
    :param transformation_steps: The transformation steps
    :param column_steps: The column steps
    """

    def __init__(self, raw_target_path: str, processed_path: str, transformation_steps: list[str], column_steps: list[str]) -> None:
        """Initialize the class.

        :param raw_target_path: The raw target path
        :param processed_path: The processed path
        :param transformation_steps: The transformation steps
        :param column_steps: The column steps
        """
        self.raw_target_path = raw_target_path
        self.processed_path = processed_path
        self.transformation_steps = transformation_steps
        self.column_steps = column_steps

    def get_pipeline(self) -> Pipeline:
        """Get the pipeline.

        :return: Pipeline object
        """
        # TODO(Epoch): Implement target pipeline
        return None
