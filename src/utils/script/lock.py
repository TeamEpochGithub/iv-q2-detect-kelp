"""Lock that makes sure only one instance of the script is running at a time."""
import os
import sys
import time
from types import TracebackType

from src.logging_utils.logger import logger

if sys.version_info < (3, 11):  # Self was added in Python 3.11
    from typing import Literal

    from typing_extensions import Self
else:
    from typing import Literal, Self


class Lock:
    """OS-wide lock based on a file created in the cwd."""

    lock_file = ".lock"

    def __init__(self) -> None:
        """Initialize the lock."""
        self.acquired = False
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.lock_file = f".lock_{os.environ['CUDA_VISIBLE_DEVICES'].replace(',', '_')}"

    def __enter__(self) -> Self:
        """Create the lock file."""
        logger.info("Acquiring lock")

        # Check if locked by checking if the file exists
        if os.path.exists(self.lock_file):
            logger.info("Waiting for lock to be released...")
        while os.path.exists(self.lock_file):
            time.sleep(1)

        # Acquire the lock by creating the file
        open(self.lock_file, "w").close()
        self.acquired = True
        logger.info("Lock acquired")

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> Literal[False]:
        """Remove the lock file. Always returns false, as it does not handle exceptions.

        :param exc_type: Exception type
        :param exc_val: Exception value
        :param exc_tb: Exception traceback
        :return: False, always.
        """
        if self.acquired and os.path.exists(self.lock_file):
            logger.info("Releasing lock...")
            os.remove(self.lock_file)
            logger.info("Lock released")
        return False
