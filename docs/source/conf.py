# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LMFlow'
copyright = 'LMFlow 2023'
author = 'The LMFlow Team'

import sys
import os
sys.path.insert(0,os.path.abspath('../..'))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

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

html_theme_options = {
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


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
