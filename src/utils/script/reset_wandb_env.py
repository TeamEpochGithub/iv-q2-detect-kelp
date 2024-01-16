"""Reset wandb environment variables."""
import os


def reset_wandb_env() -> None:
    """Reset wandb environment variables."""
    if "WANDB_RUN_ID" in os.environ:
        del os.environ["WANDB_RUN_ID"]
