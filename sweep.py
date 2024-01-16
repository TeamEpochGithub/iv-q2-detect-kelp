"""Module to convert wandb arguments to Hydra arguments and call the cv script."""
import subprocess
import sys

from src.logging_utils.logger import logger


def convert_args(args: list[str]) -> list[str]:
    """Convert wandb arguments to Hydra arguments.

    :param args: The arguments to convert.
    :return: The converted arguments.
    """
    # Convert wandb arguments to Hydra arguments
    return [arg.replace("--", "") for arg in args]


def sanitize_args(args: list[str]) -> list[str]:
    """Sanitize the arguments.

    :param args: The arguments to sanitize.
    :return: The sanitized arguments.
    """
    # A list of shell metacharacters that might be used in a command injection attack
    shell_metacharacters = [";", "|", "&", "<", ">", "(", ")", "$", "`", "\\", '"', "'"]
    sanitized_args = []
    for arg in args:
        if any(char in arg for char in shell_metacharacters):
            logger.warning(f"Warning: Argument {arg} contains shell metacharacters and will be ignored.")
        else:
            sanitized_args.append(arg)
    return sanitized_args


if __name__ == "__main__":
    # Get the original arguments
    original_args = sys.argv[1:]
    # Convert the arguments
    hydra_args = convert_args(original_args)
    # Sanitize the arguments
    sanitized_args = sanitize_args(hydra_args)
    # Call your original script with the sanitized arguments
    python_path = "venv/Scripts/python.exe"
    subprocess.run([python_path, "cv.py", *sanitized_args], check=False)  # noqa: S603
