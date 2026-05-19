# Sphinx configuration for the PorosityFE API reference site.
#
# Build locally with:
#
#     pip install -e ".[docs]"
#     python -m sphinx -b html docs docs/_build/html
#
# The site is also built (and deployed to GitHub Pages on master) by
# .github/workflows/docs.yml.
from __future__ import annotations

import os
import sys
from datetime import date

# Make the top-level porosity_fe_analysis module importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "PorosityFE"
author = "Rani Elhajjar"
copyright = f"{date.today().year}, {author}"

# Pull the version from the installed package metadata so the docs stay in
# sync with pyproject.toml. Fall back gracefully when the package is not
# importable (e.g. during a fresh clone before `pip install -e .`).
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("porosity-fe")
except Exception:  # pragma: no cover - fallback for an unbuilt checkout
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Generate stub .rst files for entries listed in autosummary directives.
autosummary_generate = True

# Document members in source order (matches the layout of
# porosity_fe_analysis.py: MaterialProperties, VoidGeometry, ...).
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
}

# Napoleon -- we use numpydoc style throughout the codebase.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx -- link out to numpy / scipy / matplotlib reference docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

# Don't fail a CI build over an intersphinx outage.
nitpicky = False
