# Minimal Sphinx makefile for the PorosityFE docs site.
#
# Usage:
#   make html       Build the HTML site under _build/html
#   make clean      Remove all build artefacts
#
# CI invokes `python -m sphinx -b html docs docs/_build/html` directly
# (see .github/workflows/docs.yml); this Makefile is only for convenience
# on developer machines.

SPHINXBUILD  ?= python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)" api

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
