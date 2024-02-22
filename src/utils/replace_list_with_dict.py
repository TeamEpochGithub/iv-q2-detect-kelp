"""Module containing a function to replace lists with integer index dicts for wandb purposes."""


def replace_list_with_dict(o: object) -> object:
    """Recursively replace lists with integer index dicts.

    This is necessary for wandb to properly show any parameters in the config that are contained in a list.

    :param o: Initially the dict, or any object recursively inside it.
    """
    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = replace_list_with_dict(v)
    elif isinstance(o, list):
        o = {i: replace_list_with_dict(v) for i, v in enumerate(o)}
    return o