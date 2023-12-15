class ColumnPipelineError(Exception):
    """
    ColumnPipelineError is an error that occurs when the column pipeline fails.
    :param message: The error message
    """

    def __init__(self, message: str) -> None:
        """
        ColumnPipelineError is an error that occurs when the column pipeline fails.
        :param message: The error message
        """
        self.message = message

    def __str__(self) -> str:
        """
        Stringify the error.
        :return: The error message
        """
        return self.message

    def __repr__(self) -> str:
        """
        Represent the error.
        :return: The error message
        """
        return self.message
