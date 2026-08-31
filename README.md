# kftools

[![Tests](https://github.com/kfuku52/kftools/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/kfuku52/kftools/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/kfuku52/kftools)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://www.python.org/)

`kftools` is a collection of Python utilities for evolutionary genomics,
phylogenetics, expression analysis, statistics, sequence models, and plotting.

## Installation

Python 3.10 or newer is required; CI tests Python 3.10–3.14. Install the current
default branch with Git and pip:

```bash
python -m pip install git+https://github.com/kfuku52/kftools.git
```

For local development, use Python 3.14 and the reproducible environment in the
[development guide](docs/development.md). Runtime minimum requirements remain in
[`pyproject.toml`](pyproject.toml). The package ships public type annotations and
a `py.typed` marker.

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

Input validation and missing-value handling are function-specific; see
[data semantics](docs/data-semantics.md). Invalid values usually raise
`ValueError`, while unsupported tree input types can raise `TypeError` and
unimplemented F1X4 operations raise `NotImplementedError`. Missing reference
species and ambiguous taxonomy matches emit `RuntimeWarning`.

## Usage and development

- [Usage examples](docs/examples.md): expression, trees, species labels, ancestor lookup, sequences, statistics, and plotting.
- [Data and mutation semantics](docs/data-semantics.md): defaults, missing values, supported models, and tree changes.
- [Input file formats](docs/file-formats.md): OU tables, FASTA, IQ-TREE checkpoints, and other logs.
- [Changes in 0.6.2](docs/changes-0.6.2.md): documentation corrections and complete source archives.
- [Development checks](docs/development.md): minimum/latest environments, typing, wheel checks, and CI.

## License

This project is licensed under the [MIT License](LICENSE).
