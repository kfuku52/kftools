# kftools

[![Tests](https://github.com/kfuku52/kftools/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/kfuku52/kftools/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/kfuku52/kftools)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue)](https://www.python.org/)

`kftools` is a collection of Python utilities for evolutionary genomics,
phylogenetics, expression analysis, statistics, sequence models, and plotting.

![kfplot examples: density scatter, cumulative histograms with boxplots, and stacked bars](docs/images/kfplot_examples.png)

## Installation

Python 3.10 or newer is required; CI tests Python 3.10–3.14. Install the current
default branch with Git and pip in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install git+https://github.com/kfuku52/kftools.git
```

The activation command above is for a Unix-style shell. On Windows PowerShell,
use `.venv\Scripts\Activate.ps1`. Pip installs the Python runtime dependencies;
ETE4 may require a compiler if no compatible wheel is available. This library
does not install a `kftools` command. Verify the active Python with:

```bash
python -c "from kftools.kfphylo import get_tree_height; print(get_tree_height('(A:1,B:1);'))"
```

The result is `1.0`. Continue with the [self-contained Python examples](docs/examples.md).
Notung, IQ-TREE, and mapNH are not invoked by the library: the corresponding
helpers read existing outputs or construct parameter strings. NCBI annotation
has separate [database prerequisites](docs/data-semantics.md#species-parsing-and-taxonomy).

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
- [Changes in 0.6.6](docs/changes-0.6.6.md): numerical precision, species mapping IDs, and input validation fixes.
- [Development checks](docs/development.md): minimum/latest environments, typing, wheel checks, and CI.

## License

This project is licensed under the [MIT License](LICENSE).
