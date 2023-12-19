class FeaturePipelineError(Exception):
    """
    FeaturePipelineError is an error that occurs when the feature pipeline fails.
    :param message: The error message
    """

    def __init__(self, message: str) -> None:
        """
        FeaturePipelineError is an error that occurs when the feature pipeline fails.
        :param message: The error message
        """
        self.message = message

    def __repr__(self) -> str:
        """
        Represent the error.
        :return: The error message
        """
        return self.message
