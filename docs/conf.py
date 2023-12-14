import importlib
import inspect
import os
import sys
from types import ModuleType
from typing import Any

# TODO Replace this with GitHub URL later
REPO_URL: str = "https://gitlab.ewi.tudelft.nl/dreamteam-epoch/epoch-iv/q2-detect-kelp/-/blob/main/"

sys.path.insert(0, os.path.abspath('../..'))

project: str = 'Detect Kelp'
copyright: str = '2024, Team Epoch.'
author: str = 'Team Epoch'

source_suffix: dict[str, str] = {'.rst': 'restructuredtext'}
root_doc: str = 'index'

extensions: list[str] = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'myst_parser',
                         'sphinx.ext.linkcode', 'sphinx.ext.viewcode',  # TODO Remove 'sphinx.ext.viewcode' once we publish our code on GitHub
                         'sphinxawesome_theme.highlighting', "sphinx_autodoc_typehints"]
autosummary_generate: bool = True
autodoc_typehints: str = "signature"

autodoc_default_options: dict[str, bool | str] = {
    'members': True,
    'undoc-members': True,
    'member-order': 'bysource',
}

viewcode_line_numbers: bool = True


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Determine the URL corresponding to the Python object.

    This is used by sphinx.ext.linkcode to generate links to our source code.

    The code for getting the line numbers is copied from https://github.com/python-websockets/websockets/blob/main/docs/conf.py.

    :param domain: domain of the object
    :param info: information about the object
    :return: URL to the object or None if it is not Python code
    """
    if domain != 'py':
        return None
    if not info['module']:
        return None

    module: ModuleType = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        obj_name, attr_name = info["fullname"].split(".")
        obj: Any = getattr(module, obj_name)
        try:
            # Object is a method of a class
            obj = getattr(obj, attr_name)
        except AttributeError:
            # Object is an attribute of a class
            return None
    else:
        obj = getattr(module, info["fullname"])

    try:
        lines: tuple[list[str], int] = inspect.getsourcelines(obj)
    except TypeError:
        # E.g. object is a typing.Union
        return None

    start: int = lines[1]
    end: int = lines[1] + len(lines[0]) - 1

    filename: str = info['module'].replace('.', '/')
    return f"{REPO_URL}{filename}.py#L{start}-L{end}"


pygments_style: str = 'sphinx'

templates_path: list[str] = ['_templates']
exclude_patterns: list[str] = ['_build', 'Thumbs.db', '.DS_Store']

html_theme: str = 'sphinxawesome_theme'
html_theme_options: dict[str, str] = {
    "logo_light": "./_static/images/logo/Epoch_Icon_Dark.png",
    "logo_dark": "./_static/images/logo/Epoch_Icon_Light.png"
}
html_favicon: str = "./_static/images/logo/Epoch_Icon_Light.png"
html_static_path: list[str] = ['_static']
html_use_smartypants: bool = True
html_show_sourcelink: bool = True
html_show_sphinx: bool = True
html_show_copyright: bool = True
