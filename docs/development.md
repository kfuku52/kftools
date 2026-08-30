# Development and verification

Use Python 3.14 for development and type checking. Runtime support remains
Python 3.10–3.14. No upper bounds or exact runtime dependency pins were added.

```sh
python3.14 -m venv .venv
make install PYTHON=.venv/bin/python
make check PYTHON=.venv/bin/python
```

`make install` uses the cross-platform dependency snapshot in
[`constraints/development-python314.txt`](../constraints/development-python314.txt).
It pins development tools and their resolved dependencies separately from the
published library requirements. Refresh it deliberately with `make lock-dev`
(requires [uv](https://docs.astral.sh/uv/pip/compile/)), then run the checks below.
For diagnosing upstream updates, `check-latest` always resolves dependencies
afresh without this snapshot.

All Makefile tools run through `$(PYTHON) -m ...`, including Ruff, mypy,
coverage, build, Twine, and pip-audit. Set `PYTHON` to an interpreter path rather
than relying on whichever executable happens to be first on PATH. `make format`,
`make lint`, `make typecheck`, and `make test` provide shorter feedback loops.

## Clean environments

```sh
make check-minimum PYTHON_MINIMUM=python3.10
make check-latest PYTHON_LATEST=python3.14
make build PYTHON=.venv/bin/python
make wheel-smoke PYTHON=.venv/bin/python
make audit PYTHON=.venv/bin/python
```

The minimum, latest, and wheel targets share
[`scripts/check_environment.py`](../scripts/check_environment.py). Each creates
a temporary virtual environment, installs from scratch, runs the checks, runs
`pip check`, and removes the environment even after failure. Interpreter paths
can be absolute. Minimum checks require 3.10; latest quality checks require
3.14. `wheel-smoke` uses `PYTHON` and imports the installed wheel outside the
repository with `PYTHONPATH` cleared. Keep one wheel in `dist/`, or select an
artifact explicitly:

```sh
python scripts/check_environment.py wheel --python python3.12 --wheel dist/kftools-0.6.0-py3-none-any.whl
```

The minimum dependency set stays in
[`constraints/minimum-python310.txt`](../constraints/minimum-python310.txt).
Missing compiler/system libraries during a first ETE4 build are installation
failures, not test failures; inspect the installation log first.

## What the checks cover

`make check` runs lint, formatting, type checks, and the warning-strict test
suite with branch coverage (minimum 85%). Numerical regression tests compare
plotted values, preserve table columns and identifiers, exercise deep trees,
and check invariants such as renaming input columns and parsing displayed names.

The type checker now reads NumPy, Matplotlib, and pandas-stubs rather than
disabling site-package type information. Its target is Python 3.14 because
current dependency stubs use newer syntax. Ruff's Python 3.10 target and runtime
tests separately enforce compatibility. ETE4, SciPy, and statsmodels imports
remain explicitly exempt from missing-stub errors; their untyped APIs still
limit static checking. Validation at runtime remains necessary.
[`tests/typing/public_api.py`](../tests/typing/public_api.py) checks consumer
return types and invalid calls; an unexpectedly unused error suppression fails
the type check, detecting accidental regressions to `Any`.

## CI and caches

CI retains Python 3.10–3.14, minimum dependencies, latest quality checks,
sdist/wheel validation, an installed-wheel smoke test, and a strict dependency
audit. Packaging and auditing share the 3.12 compatibility job, reducing the
workflow from eight jobs to six. Pushes to the default branches, pull requests,
weekly runs, and manual runs retain the existing triggers and cancellation of
superseded work. No branch protection settings are changed.

The shared [Python action](../.github/actions/python/action.yml) caches pip
downloads and built wheels by OS, architecture, CPython major/minor, and
dependency set. Restore prefixes allow reuse after Python patch or metadata
changes; weekly snapshots allow newly built wheels to enter the cache even when
requirements have not changed. Minimum and latest sets have separate primary
keys but can reuse compatible cached wheels. Pip still resolves requirements;
a cache is not a dependency lock.

A cold run still builds ETE4 when no compatible wheel is available. GitHub can
evict caches after seven days without access, so a weekly-only repository may
still have cold runs. More frequent scheduled runs were not added just to retain
cache entries. See the [GitHub cache documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
for cache scope and eviction rules. Compare installation and test steps
separately, and report whether each run restored a cache.

## Tree benchmarks

```sh
python benchmarks/tree_transfers.py --shape balanced --leaves 1000
python benchmarks/tree_transfers.py --shape comb --leaves 1000 --operation transfer_root
python benchmarks/tree_transfers.py --source /path/to/old/checkout --shape star --leaves 1000
```

Run each configuration in its own process with the same interpreter and
dependencies. The script reports all samples, median wall time, peak process
RSS, deterministic output hashes, and recursion failures; it also checks input
preservation. Timing excludes construction and serialization; peak RSS includes
imports and input construction. Results from this change are in
[`benchmarks.md`](benchmarks.md).
