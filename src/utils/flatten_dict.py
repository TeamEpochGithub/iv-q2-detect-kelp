"""Flatten a nested dictionary to be used for sklearn pipeline parameters."""
from typing import Any


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "__") -> dict[str, str]:
    """Flatten a nested dictionary to be used for sklearn pipeline parameters.

    :param d: dictionary to flatten
    :param parent_key: parent key
    :param sep: separator
    :return: flattened dictionary
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
