# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#sys.path.insert(0, os.path.abspath('.'))
#sys.path.insert(0, os.path.abspath('..VIA'))
#sys.path.insert(0, os.path.abspath('../../VIA'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.append('../VIA/')
#sys.path.insert(0, os.path.abspath('..VIA'))
#root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#sys.path.insert(0, root_dir)


# -- Project information -----------------------------------------------------

project = 'pyvia'
copyright = '2022, shobana stassen'
author = 'shobana stassen'

# The full version, including alpha/beta/rc tags
#release = 'pyvia 2022'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinxemoji.sphinxemoji', 'sphinx_rtd_theme','nbsphinx',"sphinx.ext.autodoc",'sphinx.ext.githubpages',
              'sphinx.ext.viewcode','sphinx.ext.napoleon','sphinx_autodoc_typehints','myst_nb',"sphinx_tippy"]

# Generate the API documentation when building
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_member_order = "alphabetical"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# myst
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 2


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'sphinx_rtd_theme'
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
#source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'
