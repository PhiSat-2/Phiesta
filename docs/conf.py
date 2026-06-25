# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Phiesta'
copyright = '2026, Malo de Pastor'
author = 'Malo de Pastor'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]


html_theme = "alabaster"

autosummary_generate = True
autodoc_member_order = "bysource"
autoclass_content = "both"

# À utiliser seulement si certains imports cassent la build
# autodoc_mock_imports = ["torch", "rasterio", "tifffile", "matplotlib"]