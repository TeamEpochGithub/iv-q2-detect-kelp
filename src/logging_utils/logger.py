"""Setup the logger."""
import logging
import sys
from types import TracebackType

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)


def log_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None) -> None:
    """Log any uncaught exceptions except KeyboardInterrupts.

    Based on https://stackoverflow.com/a/16993115.

    :param exc_type: The type of the exception.
    :param exc_value: The exception instance.
    :param exc_traceback: The traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("A wild %s appeared!", exc_type.__name__, exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = log_exception
