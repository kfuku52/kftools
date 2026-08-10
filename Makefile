PYTHON ?= python

.PHONY: install format lint typecheck test coverage check build audit clean

install:
	$(PYTHON) -m pip install -e '.[dev]'

format:
	ruff check --select I --fix .
	ruff format .

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy kftools

test:
	$(PYTHON) -m pytest -q

coverage:
	coverage erase
	coverage run -m pytest -q
	coverage report

check: lint typecheck coverage

build:
	$(PYTHON) -m build
	twine check dist/*

audit:
	pip-audit --strict .

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in map(Path, ('build', 'dist', 'kftools.egg-info'))]; Path('.coverage').unlink(missing_ok=True)"
