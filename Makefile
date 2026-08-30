PYTHON ?= python
PYTHON_MINIMUM ?= python3.10
PYTHON_LATEST ?= python3.14
UV ?= uv

.PHONY: install format lint typecheck test coverage check check-minimum check-latest wheel-smoke lock-dev build audit clean

install:
	$(PYTHON) -m pip install -c constraints/development-python314.txt -e '.[dev]'

format:
	$(PYTHON) -m ruff check --select I --fix .
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy kftools tests/typing

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m pytest -q
	$(PYTHON) -m coverage report

check: lint typecheck coverage

check-minimum:
	$(PYTHON) scripts/check_environment.py minimum --python "$(PYTHON_MINIMUM)"

check-latest:
	$(PYTHON) scripts/check_environment.py latest --python "$(PYTHON_LATEST)"

wheel-smoke:
	$(PYTHON) scripts/check_environment.py wheel --python "$(PYTHON)"

lock-dev:
	$(UV) pip compile --universal --python-version 3.14 --extra dev pyproject.toml -o constraints/development-python314.txt

build:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

audit:
	$(PYTHON) -m pip_audit --strict .

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in map(Path, ('build', 'dist', 'kftools.egg-info'))]; Path('.coverage').unlink(missing_ok=True)"
