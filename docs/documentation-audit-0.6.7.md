# Documentation audit for 0.6.7

Audited on 2026-09-22, starting from `master` at
`220917afd488cb2efd271b9279f20891c2f4ac74` (0.6.6). The only pre-existing
uncommitted item was an untracked `.DS_Store`, which was preserved.
The 0.6.7 version bump is required by repository push policy; this audit changes
no runtime behavior, dependencies, or scientific defaults.

## Scope and evidence

Prioritized installation, all eight runnable Python blocks in
[examples](examples.md) and [file formats](file-formats.md), the public-module
map, file readers, node/OU outputs, sequence models, and statistical defaults.
Compared signatures, processing code, and relevant existing tests, including
`test_kfog.py`, `test_kfseq.py`, `test_kfstat.py`, `test_improvements.py`, and
`test_review_regressions.py`. Inspected `pyproject.toml`, `Makefile`, the
check-environment script, and CI configuration. There is no installed CLI,
application configuration file, or API environment-override hierarchy.
The environment-check script's help was executed and compared with its parser
and Make targets. Historical release notes and benchmark reports were preserved.

## A: corrected documentation

| Location and original issue | Evidence | Correction |
| --- | --- | --- |
| [README installation](../README.md#installation): installation did not isolate dependencies or provide an immediate import check; external executable prerequisites were unclear | `pyproject.toml` dependencies and lack of script entry points; public readers; isolated installation | Add venv setup, a height calculation printing `1.0`, and clarify that log/model helpers do not invoke upstream tools. Link NCBI prerequisites. |
| [Sequence models](data-semantics.md#sequence-frequencies-and-theta-parameters) and `kfseq` docstrings: “one dictionary” obscured the list wrapper and mixed input and return types | `codon2nuc_freqs`, `get_mapnh_thetas`, `test_kfseq`, `test_improvements` and a synthetic F1X4 call | State list/tuple shapes, string return, required `thetas` argument, and the unusable empty model default. Clarify that F1X4 FASTA rejection precedes alphabet/codon validation. |
| [Tree semantics](data-semantics.md#trees-labels-and-mutation): age origin and units were unspecified | `get_tree_height`, `_node_age_from_children`, `_nwk_age_values`; multifurcating age tests and a two-tip example | State tip age zero and retention of input branch-length units. No scientific unit is inferred. |
| [Table semantics](data-semantics.md#node-tables-and-ancestor-lookup), [OU format](file-formats.md#ou-node-tables), and [plot example](examples.md#regression-and-stacked-bars): saving, relative paths, output shape, and reruns were not fully explained | Table return statements, file opens, and the example's `savefig`; exact example execution | Explain in-memory tables, no automatic result files, node IDs, plot replacement on rerun, and temporary-file removal. Add OU shape/column assertions. |
| [OU comparisons](file-formats.md#ou-node-tables): maximum-mu differences lacked an explicit direction | `_add_ou_shift_values` and OU shift tests | Define focal-node minus comparison-node maximum on the existing log scale. |
| [FASTA outputs](file-formats.md#fasta): column names lacked count definitions | `get_aln_stats`, its tests, and a six-column gapped fixture | Define alignment columns, FASTA record count, and ungapped character counts. |
| [Brunner–Munzel results](data-semantics.md#brunnermunzel-results): confidence-level default and degrees-of-freedom interpretation were omitted | `bm_test`, `_validate_bm_options`, and `test_kfstat.py` | State `alpha=0.05`, its open interval, and approximate t degrees of freedom. |
| [Wheel verification](development.md#clean-environments): explicit selection showed only `--wheel`, not a full command | `scripts/check_environment.py --help`, parser, and wheel-mode execution | Show the complete command and mark the path as a placeholder. |

`tests/test_documentation.py` executes the actual guide blocks, without copying
the examples into tests. It checks assertions, output file names, and a
nonuniform 1350 × 600 PNG. The documentation remains Markdown; there is no
separate documentation generator/build configured.

## B: unresolved reader error

**Resolved in 0.6.8:** `get_iqtree_model_stats` now wraps `EOFError` in
`ValueError`, preserving the original exception as its cause. Regression tests
cover truncated headers, compressed payloads, and trailers. The account below
records the original 0.6.7 finding.

[File formats](file-formats.md#iq-tree-and-other-logs) promises `ValueError` for
read errors. A truncated gzip checkpoint instead raises `EOFError` from
[`kfog.get_iqtree_model_stats`](../kftools/kfog.py). Its handler catches `UnicodeDecodeError` and
`OSError`, but not `EOFError`. The existing [`test_kfog_iqtree_stats`](../tests/test_kfog.py) covers a
valid gzip, non-gzip data, and invalid UTF-8, not a truncated stream.

Reproduced with the following small local input:

```python
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

from kftools.kfog import get_iqtree_model_stats

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "truncated.model.gz"
    path.write_bytes(gzip.compress(b"best_model_AIC: GTR\n")[:-4])
    get_iqtree_model_stats(path)
```

Actual exception: `EOFError: Compressed file ended before the end-of-stream
marker was reached`. Applications catching the documented `ValueError` will
not catch this failure. The implementation and its stated error contract are
left unchanged; the guide links this outstanding discrepancy. A follow-up
implementation fix should normalize this read error and add a truncated-stream
regression test.

## C and limits

No specific contradictory scientific specification was established within this
scope. Biological validity of inverse-branch weighting, expression scales, and
OU assumptions was not independently established; existing definitions were
preserved. The audit is not a proof of all public APIs or numeric edge cases.

No real NCBI database initialization, external Notung/IQ-TREE/mapNH run,
large-data analysis, dependency upgrade, or benchmark was performed. Parser
fixtures verify extraction, not compatibility with every upstream version.
Python 3.10–3.13, Windows/PowerShell, and Linux execution were not repeated;
compatibility statements were compared with metadata and CI, not certified by
this macOS Python 3.14.7 run. External URL availability and third-party scientific
claims were not independently checked. Local Markdown paths/anchors were checked.

## Execution record

The verification results below distinguish exact examples from substituted
installation inputs. All generated examples and installation/build work use
temporary directories; the existing development environment was not reinstalled.

| Command / procedure | Result and scope |
| --- | --- |
| `.venv/bin/python --version`; `.venv/bin/python -m pip check` | Python 3.14.7; no broken requirements in the existing development venv. |
| Fresh temporary venv, then `source .venv/bin/activate` and `python -m pip install git+https://github.com/kfuku52/kftools.git` | Exact README activation/install commands succeeded. Venv creation explicitly used `python3.14` instead of `python`. The remote still pointed to the starting 0.6.6 commit at verification time; this is not a Git-install test of the pending 0.6.7 commit. |
| `python -c "from kftools.kfphylo import get_tree_height; print(get_tree_height('(A:1,B:1);'))"`; `python -m pip check` | Exact README smoke command printed `1.0` from the installed package outside the checkout; dependency consistency passed. |
| `python3.14 -m venv .venv`; `make install PYTHON=.venv/bin/python` | Succeeded in a temporary source copy containing the pending changes, with the existing development constraints. Source copying substituted for `git clone` so the uncommitted version could be tested. |
| `.venv/bin/python -m pytest -q tests/test_documentation.py` | 2 tests passed, executing all eight guide blocks unchanged from their current text. OU output was 3 rows × 15 columns with the expected IDs, mu columns, tau and complementarity; PNG was 1350 × 600 in the temporary working directory. |
| Inline Python synthetic parser checks | Six-column FASTA: `num_site=6`, `num_seq=2`, `len_min=4`, `len_max=6`; selected `AAATTT` gave 0.5 A/0.5 T at each position. F1X4 returned a one-element list. Three ModelFinder keys extracted correctly. Two-tip length-2 tree gave ages `[0, 0, 2]`. These are small replacement fixtures, not real upstream program runs. |
| Truncated-gzip reproducer above | Confirmed unresolved `EOFError`; counted as a failure of the documented error contract, not as a successful reader check. |
| `.venv/bin/python scripts/check_environment.py --help` | Help, mode choices, interpreter default, and wheel-selection argument matched source and Makefile. |
| `make check PYTHON=.venv/bin/python` | Ruff lint/format and mypy passed; 190 tests and 44 subtests passed; combined statement/branch coverage 87%, above the 85% gate. |
| `make build PYTHON=.venv/bin/python` | In the isolated source copy, built 0.6.7 sdist and wheel; Twine accepted both. Existing checkout artifacts were preserved. |
| `.venv/bin/python scripts/check_environment.py wheel --wheel <temporary-source>/dist/kftools-0.6.7-py3-none-any.whl` | Actual generated absolute wheel path substituted for the documented placeholder. Fresh isolated installation passed imports, version/metadata agreement, `py.typed`, numerical APIs, plotting, and `pip check`. |
| Inline local Markdown link/heading checker; `git diff --check` | Local guide/README/AGENTS file links and heading targets resolved; no whitespace errors. External URLs were excluded. |

Minimum/latest full dependency matrices, vulnerability audit, lock regeneration,
format rewriting, and cleanup targets were not run: the task did not change
runtime requirements or check definitions, and no cleanup of existing artifacts
was needed. The clean wheel check did independently resolve runtime dependencies.
