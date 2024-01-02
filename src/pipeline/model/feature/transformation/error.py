"""Error module for the transformation pipeline."""
from src.logging_utils.logged_error import LoggedError


class TransformationPipelineError(LoggedError):
    """TransformationPipelineError is an error that occurs when the transformation pipeline fails."""
