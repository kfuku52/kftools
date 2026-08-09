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
[`setup.py`](setup.py). CI tests both those minimums and the latest resolved
versions.

## Public modules

| Module | Main functions |
| --- | --- |
| `kfexpression` | `calc_tau`, `calc_complementarity` |
| `kfseq` | `codon2nuc_freqs`, `nuc_freq2theta`, `get_mapnh_thetas`, `alignment2nuc_freqs`, `weighted_mean_root_thetas` |
| `kfphylo` | `load_phylo_tree`, `get_tree_height`, `transfer_internal_node_names`, `fill_internal_node_names`, `add_numerical_node_labels`, `transfer_root`, `check_ultrametric`, `taxonomic_annotation` |
| `kfspecies` | `SpeciesParseResult`, `parse_species_label` |
| `kfog` | Tree tables, orthogroup/node statistics, OU/regime parsers, alignment and reconciliation statistics |
| `kfstat` | `bm_test`, `brunner_munzel_test` |
| `kfplot` | `stacked_barplot`, `density_scatter`, `hist_boxplot`, `ols_annotations` |
| `kfutil` | Dictionary and RGB helpers |

Functions validate malformed, missing, non-finite, and biologically invalid
inputs and raise `ValueError` with a description of the rejected field.

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

![kfplot examples](docs/images/kfplot_examples.png)

## Data and mutation semantics

- Expression profiles and nucleotide/theta frequencies must be non-negative.
- F3X4 sequence inputs must contain complete codons and therefore have lengths
  divisible by three.
- FASTA inputs passed to `get_aln_stats` must be aligned: every sequence must
  have the same gapped length.
- `transfer_root` returns a rerooted deep copy and never mutates its input tree.
- `fill_internal_node_names`, `add_numerical_node_labels`, and `nwk2table`
  attach attributes to a supplied ETE tree object. Copy the tree first when the
  original must remain untouched.
- `taxonomic_annotation` requires an initialized ETE/NCBI taxonomy database.

## Development checks

```bash
ruff check .
mypy kftools
coverage run -m pytest -q
coverage report
python -m build
twine check dist/*
pip-audit
```

CI runs these checks on Python 3.10 through 3.14, verifies the declared minimum
dependency set, builds both source and wheel distributions, and audits resolved
runtime dependencies.

## License

This project is licensed under the [MIT License](LICENSE).
