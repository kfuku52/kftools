# Data and mutation semantics

These notes describe the public APIs, including defaults that affect numerical
results. See the [examples](examples.md) for runnable calls and
[file formats](file-formats.md) for inputs read from disk.

## Expression profiles

`calc_tau` returns one value per dataframe row in the original row order. Its
defaults, `unlog2=True, unPlus1=True`, interpret selected columns as
`log2(expression + 1)`: values are transformed with `2**x - 1` and clipped at
zero before calculating tau. Use `unlog2=False` for raw non-negative expression;
`unPlus1` has no effect in that mode. For `log2(expression)` without a pseudocount,
use `unlog2=True, unPlus1=False`. Missing/non-finite input and overflow during
back-transformation raise `ValueError`.

For two or more tissues, tau is `sum(1 - x_i / max(x)) / (n - 1)`. A single
tissue or an all-zero profile returns zero by convention. The dataframe is
not modified.

`calc_complementarity` flattens both inputs, requires equal non-zero numbers of
finite, non-negative values, and returns the mean of
`abs(a_i - b_i) / max(a_i, b_i)`. A pair of zeros contributes zero. Values range
from zero (identical profiles) to one; input arrays are not modified.

## Sequence frequencies and theta parameters

Model strings must contain exactly one of the case-sensitive tokens `F1X4` or
`F3X4`, optionally within a larger model string such as `GY+F3X4`.

| Function | F1X4 | F3X4 |
| --- | --- | --- |
| `codon2nuc_freqs` | One pooled A/T/C/G dictionary | Three dictionaries, one per codon position |
| `get_mapnh_thetas` | One theta dictionary, or none | Three theta dictionaries, or none |
| `alignment2nuc_freqs` | Not implemented | Three dictionaries for one FASTA sequence |
| `weighted_mean_root_thetas` | Not implemented | Three dictionaries for the root |

The two unimplemented paths raise `NotImplementedError` after validating their
inputs; they do not fall back to F3X4. `codon2nuc_freqs` accepts lowercase codons,
but every codon must contain exactly three A/T/C/G bases. Frequencies must be
finite and non-negative with a positive total.

`nuc_freq2theta` accepts a list or tuple of dictionaries with exactly the keys
`A`, `T`, `C`, and `G`. Each dictionary is normalized without changing the input.
For normalized frequencies, `theta = G + C`, `theta1 = A / (A + T)`, and
`theta2 = G / (G + C)`; either ratio is defined as 0.5 when its denominator is
zero. Theta values passed to `get_mapnh_thetas` must be finite and in `[0, 1]`.
An empty theta list produces `F1X4()` or `F3X4()`.

`weighted_mean_root_thetas` uses **inverse** lengths of the immediate root-child
branches. If any have length zero, only the zero-length children contribute,
with equal weights. Every root child needs a unique non-empty name and a
matching dictionary entry containing three sets of the same parameter keys.

## Trees, labels, and mutation

`load_phylo_tree` accepts an `ete4.PhyloTree`, a Newick string, or a text path
(`str` or `PathLike[str]`). Passing a tree returns the **same object**. A string
ending in `;` after trailing whitespace is removed is treated as Newick; use
`pathlib.Path` for a filename ending in `;` or to request a file-specific error
for a missing file. Other strings are first tried as paths, then as Newick if
the path does not exist. Unsupported types such as numbers raise `TypeError`;
`None`, empty, unreadable, or malformed tree inputs raise `ValueError`.

The default ETE parser is `1` (internal node names). `nwk2table(attr="support")`
uses parser `0` for strings and paths; an already supplied tree is not reparsed.
Distance calculations require explicit finite, non-negative non-root branch
lengths. `check_ultrametric` defaults to `tol=0`, so its comparison is exact;
pass a positive absolute tolerance when appropriate for rounded input.
`nwk2table(age=True, attr="dist")` and `node_gene2species(is_ultrametric=True)`
use exact ultrametric checks and inspect every child at multifurcations.

| Function | Effect on a supplied ETE tree |
| --- | --- |
| `load_phylo_tree` | Returns it unchanged without copying or revalidation |
| `get_tree_height`, `check_ultrametric` | Read only |
| `fill_internal_node_names` | Assigns missing internal names in place |
| `add_numerical_node_labels`, `nwk2table` | Assign or overwrite `branch_id` attributes in place |
| `taxonomic_annotation` | Writes species/taxonomy attributes in place |
| `get_misc_node_statistics` | Writes `branch_id`, `sci_name`, and `taxid` even when `tax_annot=False` |
| `transfer_root`, `transfer_internal_node_names` | Return a deep-copied target; neither input is modified |
| `node_gene2species` | Copies the gene tree for its work; neither input is modified |

Copy a tree before calling an in-place function if its original attributes must
remain untouched. A failed in-place annotation may already have changed some
attributes. Tree transfers copy parent/child links iteratively, but arbitrarily
nested user metadata still has Python's normal deepcopy limits.

`fill_internal_node_names` assigns unused `n1`, `n2`, … names in ETE traversal
order and preserves existing names. These are **not descendant-derived names**;
reordering children can change which clade receives each generated name.
`add_numerical_node_labels` instead ranks descendant-leaf signatures built from
sorted, unique, non-empty leaf names. Its `branch_id` values match the ranking
used by [CSUBST's `numerical_label`](https://github.com/kfuku52/csubst/blob/master/csubst/tree.py)
for the same labelled, rooted topology. Changing labels, rooting, or topology
can change branch IDs; branch lengths do not determine them.

Both transfer functions support multifurcations. Internal names are matched by
exact rooted clade signatures, including repeated clades at unary nodes.
Binary roots are transferred to a matching edge; roots with three or more
children require a unique existing vertex with the same incident leaf partition.
Target refinements are never collapsed or resolved implicitly. Vertex-root
transfers retain target branch lengths and pairwise tip distances. For a binary
root with a positive total source root-child length, the source length ratio
divides the target's root edge length. If both source lengths are zero, that
redistribution is skipped. Target lengths elsewhere are not replaced.

## Species parsing and taxonomy

`parse_species_label` defaults to `species_parser="legacy"`, which keeps the
first two underscore/whitespace-separated tokens. Choose `"taxonomic"` to
recognize qualifiers such as `cf.` and ranks such as `subsp.` and `strain`.
The parser returns `SpeciesParseResult` with `species_label`, `scientific_name`,
and `taxonomy_query`; the lookup name can be less specific than the display name.
For example, `Bacillus_subtilis_subsp_168_gene3` produces a subspecies label but
uses `Bacillus subtilis` for taxonomy lookup in taxonomic mode. Parsing alone
does not check that a taxon exists in NCBI.

For nonstandard gene IDs, pass a regex with named `genus`/`species` groups,
`{"type": "regex", "pattern": ..., "group": ...}`, a mapping config
`{"type": "map", "mapping": ...}`, or a callable. A callable can return a
`SpeciesParseResult`, string, genus/species pair, or dictionary. `parser` is an
alias for `species_parser`; supplying both raises `ValueError`. The same parser
options apply to tree taxonomy and gene/species mapping.

`taxonomic_annotation` calls ETE's default `NCBITaxa()` database. If the local
database is absent, ETE may download and build it; this needs network access,
time, disk space, and a writable data directory. Initialize it deliberately
before a large run, and record the database snapshot when reproducing results.
See [ETE's taxonomy documentation](https://etetoolkit.github.io/ete/reference/reference_taxonomy.html).
Missing names raise `ValueError`; multiple taxid matches emit `RuntimeWarning`
and use the first match. `get_misc_node_statistics` skips NCBI access by default
(`tax_annot=False`), filling `taxid` with `-999`.

## Node tables and ancestor lookup

Join and compare node tables by `branch_id`, not by dataframe row position.
`nwk2table` sorts rows by branch ID; other tree-to-table functions can return
traversal order. Specify an attribute explicitly, for example
`nwk2table(tree, attr="name", parent=True)`; the historical `attr=""` default
produces an empty-named, missing-value column.

| Table | Missing scalar relationships | Complete relationships |
| --- | --- | --- |
| `nwk2table(parent=True, sister=True)` | `-1` for absent parent/sister | `sisters` tuples |
| `get_misc_node_statistics` | `-999` for absent parent/sister/child | `children` and `sisters` tuples |

Relationship tuples are sorted by branch ID. Legacy `sister`, `child1`, and
`child2` columns select the first one or two IDs from those tuples, so they do
not capture every relationship at a multifurcation. Empty relationships use
`()`. Reordering children does not change these relationships when keyed by
branch ID, though it can change dataframe row order.

`node_gene2species` maps descendants to their species-tree common ancestor.
Species absent from the reference tree emit `RuntimeWarning`; affected nodes
and their ancestors have an empty `spnode_coverage` string. Optional
`spnode_age` values also use empty strings when no mapping is available.

`compute_delta` returns a copy with `delta_<column> = child - parent`, preserving
the index, row order, and unrelated columns. Branch IDs must be unique across
the supplied frame; split multiple orthogroups first if they reuse IDs. A
missing parent or value yields NaN. An existing `delta_<column>` is overwritten
in the result, and numeric text in the selected input column is converted.

`get_most_recent` and `MostRecentLookup.find` start at the requested node itself
and walk toward the root, returning the first matching row's requested value.
They return NaN if no match is found before a missing node/parent or a cycle.
Duplicate branch IDs within an orthogroup use the first row. Missing target
values can match each other. `prepare_most_recent_lookup` builds separate tables
for repeated queries; rebuild it after changing source data. These dataframe
copies do not recursively copy arbitrary Python objects stored in cells.

## Plotting and missing values

`ols_annotations` defaults to **median regression**, `method="quantreg"` at
quantile 0.5, despite its name. Set `method="ols"` for ordinary least squares.
Both include an intercept and require at least two finite x/y observations.
Default statistics are `N`, `slope`, and `slope_p`. Additional choices are
`rsquared`, `rsquared_adj` (OLS only), and `rsquared_p` (OLS model F-test p-value;
NaN for median regression). `rsquared` means ordinary OLS R² or median
regression pseudo R², with distinct labels. Undefined statistics are NaN;
constant x uses the response mean/median with an unidentified slope.

| Function | Missing/non-finite observations |
| --- | --- |
| `calc_tau`, `calc_complementarity`, `bm_test`, `ols_annotations` | Reject missing and non-finite numerical values |
| `brunner_munzel_test` | Discards masked, NaN, and infinite sample values; needs at least two retained values per sample |
| `density_scatter` | Discards x/y pairs with missing or non-finite values; requires a retained pair |
| `stacked_barplot` | Averages each component's non-missing values within a category; rejects infinities |
| `hist_boxplot` | Omits missing category/value rows from each distribution; rejects infinities |

`hist_boxplot` raises `ValueError` if an observed category has no retained values.

Stacked bars first average each component, then stack the means. Positive and
negative means accumulate separately from zero. Entirely missing components
contribute no height; categories with no observed components, missing category
labels, and unused categorical levels are omitted. Pass a list of component
columns on `y` for vertical bars or on `x` for horizontal bars; the other axis
is the category column. `colors` and `ax` are required arguments but accept
`None` for default colors and a new axis.

`density_scatter` returns an axis unless `return_ims=True`, which returns its
`AxesImage`. `hue_log=True` applies log2 to bin counts. With a GLM family whose
link is log, it fits on the original response, then plots natural-log response
values/predictions and calculates correlations on the retained transformed
data. This is separate from the color-count transformation.

## Brunner–Munzel results

`bm_test` returns `(statistic, dof, pvalue, probability_of_superiority,
confidence_low, confidence_high)`. The probability estimate is
`P(X < Y) + 0.5 * P(X = Y)`. `ttype=0` selects a two-sided test, positive values
select `X < Y`, and negative values select `X > Y`. The confidence interval is
always a two-sided `1 - alpha` t approximation and is not clipped to `[0, 1]`.
Zero pooled variance raises `ValueError`.

`brunner_munzel_test` returns only `(statistic, pvalue)` after filtering its
samples. `alternative` accepts `less`/`l`, `greater`/`g`, and `two_sided`;
`two-sided`, `two.sided`, `two sided`, and capitalization variants are accepted.

See [changes in 0.6.0](changes-0.6.0.md) for the numerical behavior corrected in
that release, and the [tree measurements](benchmarks.md) for scaling limits.
