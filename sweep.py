"""Module to convert wandb arguments to Hydra arguments and call the cv script."""
import ast
import subprocess
import sys


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_to_hydra_format(model_str: str, sep="."):
    # Remove the leading "--model=" and convert the string to a dictionary
    model_dict = ast.literal_eval(model_str.replace("--model=", "").replace("=", ":"))
    flattened_dict = flatten_dict(model_dict, sep=sep)

    hydra_args = [f"model.{k}={v!r}" for k, v in flattened_dict.items()]
    return hydra_args


if __name__ == "__main__":
    # Get the original arguments
    original_args = sys.argv[1:]
    # Convert the arguments with a custom separator (e.g., underscore)
    hydra_args = convert_to_hydra_format(original_args[0])
    hydra_args = [arg.replace("'_target_'", "_target_").replace("'sigma'", "sigma") for arg in hydra_args]
    # Turn list of strings into one string with spaces

    # Call your original script with the sanitized arguments
    python_path = "venv/Scripts/python.exe"
    subprocess.run([python_path, "cv.py", *hydra_args], check=False)  # noqa
