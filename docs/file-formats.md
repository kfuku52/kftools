# Input file formats

File-reading APIs accept text paths or `pathlib.Path`, not open file handles.
Tree APIs also accept Newick strings unless stated otherwise; see
[tree input semantics](data-semantics.md#trees-labels-and-mutation).

## OU node tables

`kfog.ou2table(regime_file, leaf_file, input_tree_file)` requires **three existing
files**. Its tree argument is a path, not inline Newick. The tree uses ETE parser
1, so internal labels are names. Non-empty node names must be unique, and every
node named in a regime assignment must exist in the tree.

The first two files are tab-separated tables with headers:

| File | Required columns | Meaning |
| --- | --- | --- |
| `regime_file` | `node_name`, `regime` | Explicit regime assignments; other nodes inherit their parent's regime |
| `leaf_file` | `node_name`, `param`, `regime`, at least one trait column | Trait estimates per regime; every additional column is treated as a trait |

Regime IDs are non-negative integers; missing IDs are permitted in table rows.
An unassigned root starts in regime 0. Rows with an assigned regime need a
non-empty `node_name` in `regime_file`. Conflicting assignments to the same node
are rejected. `ou2table` reads `node_name` and `param` as text, preserving names
such as `004` and `NA`.

For trait means, `param="expectations"` is accepted as an alias for `"mu"`.
Identical rows of regime/trait means are deduplicated, and remaining values are
averaged by regime, ignoring missing values per trait. Every regime used by the
tree needs a finite mean for every trait. Trait columns must otherwise be
numeric or missing.

The derived expression statistics assume that mu values are on the
`log2(expression + 1)` scale. `mu_<trait>` retains that scale. Tau and
complementarity use `max(2**mu - 1, 0)`; `delta_maxmu` compares the maxima of mu
on the original log scale. There is no option to select raw-expression mu.

`delta_tau` is child-minus-parent tau, with NaN at the root. A node is a shift
when its regime differs from its parent's; the root is not a shift. On shift
rows, `sister_branch_ids`, `delta_maxmu_sisters`, and
`mu_complementarity_sisters` are aligned tuples sorted by sister branch ID.
`delta_maxmu_parent` and `mu_complementarity_parent` compare with the parent.
Legacy scalar `delta_maxmu` and `mu_complementarity` refer to the sister and are
populated only when exactly one sister exists. Non-shift rows have empty sister
tuples and NaN comparison metrics. `num_child_shift` counts immediate shifted
children for internal nodes and is NaN for leaves.

This self-contained example creates temporary input files and checks a shift
from regime 0 to 1:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from kftools.kfog import ou2table

with TemporaryDirectory() as temporary:
    directory = Path(temporary)
    tree_file = directory / "tree.nwk"
    regime_file = directory / "regimes.tsv"
    leaf_file = directory / "means.tsv"
    tree_file.write_text("(A:1,B:1)Root;", encoding="utf-8")
    regime_file.write_text("node_name\tregime\nRoot\t0\nA\t1\n", encoding="utf-8")
    leaf_file.write_text(
        "node_name\tparam\tregime\tleaf\troot\nRoot\tmu\t0\t1\t1\nA\tmu\t1\t3\t1\n",
        encoding="utf-8",
    )
    nodes = ou2table(regime_file, leaf_file, tree_file)

shift = nodes.loc[nodes["is_shift"] == 1].iloc[0]
assert shift["regime"] == 1
np.testing.assert_allclose(shift["tau"], 6 / 7)
np.testing.assert_allclose(shift["mu_complementarity_parent"], 3 / 7)
```

## OU parameter summaries

Despite its name, `kfog.regime2tree(file)` returns a **dictionary**, not an ETE
tree. It reads a tab-separated table with `param`, `regime`, and trait columns;
an optional `node_name` column is excluded from traits. Rows with missing regime
IDs supply global parameters under keys such as `alpha_leaf` and `sigma2_leaf`.
Conflicting rows for the same parameter are rejected.

When both `alpha` and `sigma2` are present, it adds
`gamma_<trait> = sigma2_<trait> / (2 * alpha_<trait>)`, requiring finite numeric
values and non-zero alpha. `num_regime` is **maximum regime ID plus one**, or
zero when all IDs are missing; it is not a count of distinct IDs for sparse
numbering. This reader uses pandas' normal missing-value inference, unlike the
identifier-preserving reader in `ou2table`.

## FASTA

`kfog.get_aln_stats` reads plain-text FASTA. All sequences must have the same
gapped length. It returns `num_site`, `num_seq`, `len_max`, and `len_min`; only
`-` is excluded when counting ungapped lengths. An empty file returns zeros.
It counts characters rather than validating a nucleotide alphabet.

`kfseq.alignment2nuc_freqs` reads one named sequence from plain-text FASTA. A
name may match the entire header or its first whitespace-separated identifier;
multiple matching headers are rejected. F3X4 requires at least one complete
codon, length divisible by three, and A/T/C/G only (case-insensitive). Gaps,
ambiguity codes such as `N`, and RNA `U` are rejected. Only the selected sequence
is checked; this function does not validate alignment lengths of other records.
F1X4 is not implemented here.

## IQ-TREE and other logs

`kfog.get_iqtree_model_stats` reads a **gzip-compressed ModelFinder checkpoint**,
normally `*.model.gz`, containing `best_model_AIC:`, `best_model_AICc:`, and
`best_model_BIC:` lines. It returns the model names as `iqtree_best_AIC`,
`iqtree_best_AICc`, and `iqtree_best_BIC`. It does not parse the ordinary `.log`
or `.iqtree` report, and simply gzipping those reports does not convert their
contents to the expected format. Absent keys are omitted from the result. For
checkpoint generation, see the [IQ-TREE command reference](https://iqtree.github.io/doc/Command-Reference#model-selection).

The remaining log readers accept plain text:

| Function | Recognized content / output |
| --- | --- |
| `get_notung_root_stats` | `Number of optimal roots`, `Best rooting score`, and `worst rooting score` → `ntg_num_opt_root`, `ntg_best_root_score`, `ntg_worst_root_score` |
| `get_notung_reconcil_stats` | Notung duplication, co-divergence, transfer, loss, and polytomy counts → `ntg_num_dup`, `ntg_num_codiv`, `ntg_num_transfer`, `ntg_num_loss`, `ntg_num_polytomy` |
| `get_root_stats` | `Returning the ... tree` and `root positions with rho peak:` → `rooting_method`, `num_rho_peak` |
| `get_dating_method` | File contents with newline characters removed, without validating the method name |

The dictionary log readers return only recognized fields; an empty dictionary
does not certify a valid or successful upstream run. Check that required output
keys exist. Read errors raise `ValueError`.
