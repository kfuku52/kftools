# Development and verification

Use Python 3.14 for development and type checking. The package requires Python
3.10 or newer; CI covers 3.10–3.14 on Linux. The shell commands below assume
Git, Make, and a Unix-style shell (Linux/macOS, or an equivalent environment).
Run them from a checkout of this repository, not from an installed wheel.

```sh
git clone https://github.com/kfuku52/kftools.git
cd kftools
python3.14 -m venv .venv
make install PYTHON=.venv/bin/python
make check PYTHON=.venv/bin/python
```

`make install` uses the cross-platform dependency snapshot in
[`constraints/development-python314.txt`](../constraints/development-python314.txt).
It pins development tools and their resolved Python dependencies separately
from the published library requirements. It does not lock the interpreter,
system libraries, compilers, or isolated build dependencies. Start with a fresh
virtual environment for reproducibility: `make install` does not remove extra
packages already present. The universal snapshot includes platform markers;
it is not evidence that every platform has been tested.

`make lock-dev` regenerates the snapshot using [uv](https://docs.astral.sh/uv/pip/compile/),
which must be installed separately. Existing pins are retained when compatible;
this command alone does **not** upgrade all dependencies. To intentionally
upgrade the snapshot, run:

```sh
uv pip compile --universal --python-version 3.14 --extra dev --upgrade pyproject.toml -o constraints/development-python314.txt
```

Review the diff, reinstall into a fresh environment, and run the checks below.
For diagnosing upstream updates without changing the snapshot, `check-latest`
resolves dependencies afresh without it.

Python quality and packaging tools run through `$(PYTHON) -m ...`, including
Ruff, mypy, coverage, build, Twine, and pip-audit. Environment checks run the
shared Python script; `lock-dev` invokes `$(UV)` instead. Set `PYTHON` to the
desired interpreter. `make format`, `make lint`, `make typecheck`, and `make test`
provide shorter feedback loops; only `format` changes source formatting.

## Clean environments

```sh
make check-minimum PYTHON=.venv/bin/python PYTHON_MINIMUM=python3.10
make check-latest PYTHON=.venv/bin/python PYTHON_LATEST=python3.14
make build PYTHON=.venv/bin/python
make wheel-smoke PYTHON=.venv/bin/python
make audit PYTHON=.venv/bin/python
```

The minimum, latest, and wheel targets share
[`scripts/check_environment.py`](../scripts/check_environment.py). Each creates
a temporary virtual environment, installs from scratch, runs its checks and
then `pip check` if those succeed. Temporary environments are removed even
after failure. `PYTHON` runs the helper; `PYTHON_MINIMUM`/`PYTHON_LATEST` select
the interpreter used to create the isolated environment. These interpreters
must already be installed. Minimum checks require 3.10; latest quality checks
require 3.14.

`wheel-smoke` uses `PYTHON` and imports the installed wheel outside the repository
with `PYTHONPATH` cleared. Keep exactly one wheel in `dist/`. Repeated builds of
different versions leave multiple wheels; use `make clean PYTHON=.venv/bin/python`
before building if those artifacts can be discarded, or choose the wheel with
`--wheel /absolute/path/to/package.whl`. To check the single wheel using 3.12:

```sh
.venv/bin/python scripts/check_environment.py wheel --python python3.12
```

`make clean` removes `build/`, `dist/`, `kftools.egg-info/`, and `.coverage`.
The minimum dependency set stays in
[`constraints/minimum-python310.txt`](../constraints/minimum-python310.txt).
It fixes declared runtime lower bounds and one compatibility constraint, not
every transitive dependency or test tool. A first ETE4 installation may compile
from source; if installation fails, inspect missing compiler/system-library
diagnostics before investigating the test suite.

## What the checks cover

`make check` runs lint, format validation, type checks, and the warning-strict test
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

CI has six jobs: latest quality checks on 3.14, compatibility tests on
3.10–3.13, and minimum dependencies on 3.10. Packaging, installed-wheel smoke
testing, and a strict dependency audit share the 3.12 compatibility job. Triggers
are pushes to `master`/`main`, pull requests targeting those branches, Monday
03:23 UTC runs, and manual dispatch. Superseded runs are cancelled.

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
.venv/bin/python benchmarks/tree_transfers.py --shape balanced --leaves 1000
.venv/bin/python benchmarks/tree_transfers.py --shape comb --leaves 1000 --operation transfer_root
.venv/bin/python benchmarks/tree_transfers.py --source /path/to/old/checkout --shape star --leaves 1000
```

Run each configuration in its own process with the same interpreter and
dependencies. The script reports all samples, median wall time, peak process
RSS, deterministic output hashes, and recursion failures; it also checks input
preservation. Timing excludes construction and serialization; peak RSS includes
imports and input construction. Measurements for the 0.6.0 changes are in
[`benchmarks.md`](benchmarks.md).

## Documentation and distributions

The maintained guides live in `docs/` and the public function docstrings. Update
examples and data/file semantics together when an interface changes. Source
distributions include the guides, images, development scripts, constraints,
and tests via [`MANIFEST.in`](../MANIFEST.in); wheels include the library and
`py.typed`, with the online documentation linked in package metadata.

The version is defined in [`kftools/__init__.py`](../kftools/__init__.py).
Repository policy requires a version bump before pushing changes to GitHub,
including documentation-only changes. Historical change notes and benchmark
versions should continue to identify the release they describe.
