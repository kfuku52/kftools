# kftools

[![Tests](https://github.com/kfuku52/kftools/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/kfuku52/kftools/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/kfuku52/kftools)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://www.python.org/)

`kftools` is a collection of Python utilities for evolutionary genomics,
phylogenetics, expression analysis, statistics, sequence models, and plotting.

## Installation

Python 3.10 or newer is required.

```bash
pip install git+https://github.com/kfuku52/kftools
```

For local development:

```bash
git clone https://github.com/kfuku52/kftools.git
cd kftools
pip install -e '.[dev]'
```

Runtime dependencies and their tested minimum versions are declared in
[`pyproject.toml`](pyproject.toml). CI tests both the latest resolved versions
and the Python 3.10 minimum set recorded in
[`constraints/minimum-python310.txt`](constraints/minimum-python310.txt).
The package ships a `py.typed` marker, so type checkers can consume its public
annotations.

## Public modules

| Module | Main functions |
| --- | --- |
| `kfexpression` | `calc_tau`, `calc_complementarity` |
| `kfseq` | `codon2nuc_freqs`, `nuc_freq2theta`, `get_mapnh_thetas`, `alignment2nuc_freqs`, `weighted_mean_root_thetas` |
| `kfphylo` | `load_phylo_tree`, `get_tree_height`, `transfer_internal_node_names`, `fill_internal_node_names`, `add_numerical_node_labels`, `transfer_root`, `check_ultrametric`, `taxonomic_annotation` |
| `kfspecies` | `SpeciesParseResult`, `parse_species_label` |
| `kfog` | Tree tables, orthogroup/node statistics, reusable ancestor lookup, OU/regime parsers, alignment and reconciliation statistics |
| `kfstat` | `bm_test`, `brunner_munzel_test` |
| `kfplot` | `stacked_barplot`, `density_scatter`, `hist_boxplot`, `ols_annotations` |
| `kfutil` | Dictionary and RGB helpers |

Functions validate malformed, missing, non-finite, and biologically invalid
inputs and raise `ValueError` with a description of the rejected field.
Recoverable data gaps, such as gene species absent from a reference species
tree, are reported with Python warnings and can be filtered or captured with
the standard `warnings` module.

## Examples

Calculate the standard tissue-specificity tau index:

```python
import pandas as pd

from kftools.kfexpression import calc_tau

expression = pd.DataFrame(
    {
        "leaf": [10.0, 5.0],
        "root": [0.0, 5.0],
    }
)
tau = calc_tau(expression, ["leaf", "root"], unlog2=False, unPlus1=False)
# array([1., 0.])
```

Read and inspect a phylogenetic tree:

```python
from kftools.kfphylo import check_ultrametric, get_tree_height

tree = "((A:1,B:1):2,C:3);"
assert check_ultrametric(tree)
assert get_tree_height(tree) == 3.0
```

Parse a taxonomic label:

```python
from kftools.kfspecies import parse_species_label

species = parse_species_label(
    "Bacillus_subtilis_subsp_168_gene3",
    species_parser="taxonomic",
)
assert species.scientific_name == "Bacillus subtilis subsp. 168"
```

Prepare a lookup once when many nearest-ancestor queries use the same table:

```python
import pandas as pd

from kftools.kfog import prepare_most_recent_lookup

branch_table = pd.DataFrame(
    {
        "orthogroup": ["og1", "og1"],
        "branch_id": [0, 1],
        "parent": [1, 1],
        "is_shift": [0, 1],
        "regime": [0, 1],
    }
)
lookup = prepare_most_recent_lookup(
    branch_table,
    target_col="is_shift",
    return_col="regime",
)
regime = lookup.find(0, "og1", target_value=1)
assert regime == 1
```

![kfplot examples](docs/images/kfplot_examples.png)

## Data and mutation semantics

- Expression profiles and nucleotide/theta frequencies must be non-negative.
- F3X4 sequence inputs must contain complete codons and therefore have lengths
  divisible by three.
- FASTA inputs passed to `get_aln_stats` must be aligned: every sequence must
  have the same gapped length.
- `transfer_root` and `transfer_internal_node_names` return deep-copied results
  and never mutate their input trees.
- Both transfer functions support multifurcations. Internal names are matched
  by exact rooted clade signatures. Binary roots are transferred to a matching
  edge; roots with three or more children are transferred only when the target
  already contains a unique vertex with the same incident leaf partition.
  Target refinements are never collapsed or resolved implicitly. Vertex-root
  transfers retain the target branch lengths and all pairwise tip distances;
  source root-child lengths affect only binary edge-root placement.
- `fill_internal_node_names`, `add_numerical_node_labels`, and `nwk2table`
  attach attributes to a supplied ETE tree object. Copy the tree first when the
  original must remain untouched.
- With `sister=True`, `nwk2table` returns the legacy scalar `sister` column and
  a lossless tuple-valued `sisters` column. `get_misc_node_statistics` likewise
  exposes `children` and `sisters` tuples while retaining `child1`, `child2`,
  and `sister` for compatibility. Relationship tuples and their legacy scalar
  projections are sorted by deterministic `branch_id`, so input child order
  does not change the result.
- Ultrametric age calculations inspect every child at a multifurcation and
  reject inconsistent child-derived ages rather than selecting one child.
- On OU shift rows, `ou2table` records every sister in the aligned,
  branch-ID-sorted tuple columns `sister_branch_ids`, `delta_maxmu_sisters`,
  and `mu_complementarity_sisters`. `delta_maxmu_parent` and
  `mu_complementarity_parent` compare the shift with its parent. The legacy
  scalar sister columns remain populated for exactly one sister and are `NaN`
  for multifurcations, avoiding an arbitrary first-sister result.
- `taxonomic_annotation` requires an initialized ETE/NCBI taxonomy database.

## Development checks

```bash
make install
make check
make build
make audit
```

`make check` checks formatting and lint rules, type-checks the package,
and runs the warning-strict test suite with branch coverage. Individual targets
(`format`, `lint`, `typecheck`, `test`, and `coverage`) are available for a
shorter edit-feedback loop.

CI runs compatibility tests on Python 3.10 through 3.13 and the full quality
suite on Python 3.14. It also verifies the Python 3.10 minimum dependency set,
builds source and wheel distributions, smoke-tests the installed wheel outside
the repository, and audits resolved runtime dependencies.

## License

This project is licensed under the [MIT License](LICENSE).
