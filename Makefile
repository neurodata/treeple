# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= pytest

all: clean

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	$(PYTHON) setup.py clean
	rm -rf _build
	rm -rf _build
	rm -rf dist
	rm -rf oblique_forests.egg-info

clean-ctags:
	rm -f tags
	rm junit-results.xml

clean: clean-build clean-so clean-ctags

build-dev:
	pip install --verbose --no-build-isolation --editable .

build-docs:
	@echo "Building documentation"
	make -C docs/ clean
	make -C docs/ html-noplot
	cd docs/ && make view

build-pipy:
	python setup.py sdist bdist_wheel

test-pipy:
	twine check dist/*
	twine upload --repository testpypi dist/*
