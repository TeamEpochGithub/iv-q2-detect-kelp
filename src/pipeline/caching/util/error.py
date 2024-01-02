"""Error module for the caching pipeline."""
from src.logging_utils.logged_error import LoggedError


class CachePipelineError(LoggedError):
    """CachePipelineError is an error that occurs when the cache pipeline fails."""
