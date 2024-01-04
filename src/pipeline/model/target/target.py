"""TargetPipeline class sets up the target pipeline."""

from sklearn.pipeline import Pipeline


class TargetPipeline(Pipeline):
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

    # Make this the same way as the other steps with _get_steps() and super().__init__(self._get_steps()) in the init
    # Important the name of the steps can not be the same as any of the arguments of the init. sklearn will throw an error

    def get_pipeline(self) -> Pipeline:
        """Get the pipeline.

        :return: Pipeline object
        """
        # TODO(Epoch): Implement target pipeline
        return None
