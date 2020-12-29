import os
import sys

sys.path.insert(0, os.path.abspath('./..'))

# -- Project information -----------------------------------------------------
project = 'Skeltorch'
copyright = '2021, David Álvarez de la Torre'
author = 'David Álvarez de la Torre'

# -- General configuration ---------------------------------------------------
extensions = ['recommonmark', 'autoapi.extension', 'sphinx.ext.napoleon']
master_doc = 'index'
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Auto-Doc ----------------------------------------------------------------
autoapi_dirs = ['../skeltorch']
autoapi_generate_api_docs = False
