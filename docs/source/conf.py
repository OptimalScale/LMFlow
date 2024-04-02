# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LMFlow'
copyright = 'LMFlow 2024'
author = 'The LMFlow Team'

import sys
import os
sys.path.insert(0,os.path.abspath('../..'))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = []

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    'myst_parser',
    'autoapi.extension',
    #"sphinxext.rediraffe",
    "sphinx_design",
    #"sphinx_copybutton",
    # For extension examples and demos
    #"ablog",
    "matplotlib.sphinxext.plot_directive",
    #"myst_nb",
    # "nbsphinx",  # Uncomment and comment-out MyST-NB for local testing purposes.
    "numpydoc",
    #"sphinx_togglebutton",
    #"sphinx_favicon",
]

autosummary_generate = True

autoapi_type = 'python'
autoapi_dirs = ['../../src']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = []
# html_logo = "_static/logo.png"
html_theme_options = {
    "announcement": "We've released our memory-efficient finetuning algorithm LISA, check out [<a href='https://arxiv.org/pdf/2403.17919.pdf'>Paper</a>][<a href='https://github.com/OptimalScale/LMFlow#finetuning-lisa'>User Guide</a>] for more details!",
    "back_to_top_button": False,
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "LMFlow",
            "url": "https://github.com/OptimalScale/LMFlow",
            "icon": "_static/logo5.svg",
            "type": "local",
            "attributes": {"target": "_blank"},
        },
    ],
    "logo": {
        "text": "LMFlow",
        "image_dark": "_static/logo5.svg",
        "alt_text": "LMFlow",
    },
   }

