# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAUTOGEN = sphinx-autogen
BUILDDIR      = _build

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees  $(SPHINXOPTS) .

.PHONY: help clean html linkcheck doctest api

all: html

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  all        generate the complete webpage"
	@echo "  html       make only the HTML files from the existing rst sources"
	@echo "  linkcheck  check all external links for integrity"
	@echo "  doctest    run all doctests embedded in the documentation (if enabled)"

clean:
	rm -rf $(BUILDDIR)/html/*
	rm -rf $(BUILDDIR)/doctrees
	rm -rf $(BUILDDIR)/linkcheck
	rm -rf api/generated
	rm -rf .ipynb_checkpoints
	rm -rf *.h5

api:
	@echo
	@echo "Building API docs."
	@echo
	$(SPHINXAUTOGEN) -i -t _templates -o api/generated api/index.rst

html: api
	@echo
	@echo "Building HTML files."
	@echo
	# Set PYGMT_USE_EXTERNAL_DISPLAY to "false" to disable external display
	PYGMT_USE_EXTERNAL_DISPLAY="false" $(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

linkcheck: api
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."

serve:
	cd $(BUILDDIR)/html && python -m http.server 8001
