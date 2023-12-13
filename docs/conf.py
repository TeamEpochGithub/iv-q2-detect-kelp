# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Detect Kelp'
copyright = '2024, Team Epoch'
author = 'Team Epoch'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary',
              'sphinxawesome_theme.highlighting', "sphinx_autodoc_typehints", 'myst_parser']
autosummary_generate = True
autodoc_typehints = "signature"

autodoc_default_options = {
    'members':           True,
    'undoc-members':     True,
    'member-order':      'bysource',
}

source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinxawesome_theme'
html_theme_options = {
    "logo_light": "./_static/images/logo/Epoch_Icon_Dark.png",
    "logo_dark": "./_static/images/logo/Epoch_Icon_Light.png"
}
html_favicon = "./_static/images/logo/Epoch_Icon_Light.png"
html_static_path = ['_static']
html_use_smartypants = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
