"""Module for an exception (base class) that logs itself when raised."""
from src.logging_utils.logger import logger


class LoggedError(Exception):
    """Exception that logs itself when raised."""

    def __init__(self, message: str) -> None:
        """Exception that logs itself when raised.

        :param message: The error message
        """
        logger.error(message)
        super().__init__(message)
