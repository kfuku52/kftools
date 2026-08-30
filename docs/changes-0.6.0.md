# Changes in 0.6.0

All existing public module import paths remain available. This release corrects
results and validation for the cases below; downstream comparisons or snapshots
that relied on the old results should be updated.

- Constant predictors are handled explicitly by regression annotations.
  Unidentified slopes and inference are NaN; OLS draws the response mean and
  median regression draws its median. This works with statsmodels 0.15 as well
  as the declared minimum. Constant responses with varying predictors have
  slope zero and undefined R²/inference.
- Regression uses internal numeric design columns, so a predictor named
  `const` behaves like any other name. Centering/scaling avoids avoidable loss
  of rank for large predictor offsets. Unexpected model failures are reported,
  rather than silently replaced with a median line.
- `rsquared` now displays ordinary OLS R². Request `rsquared_adj` for adjusted
  R². Median regression displays `pseudo R2`; adjusted R² is OLS-only. An OLS
  two-point fit may have ordinary R² while adjusted R² and p-values remain NaN.
- Stacked bars first average each component over its non-missing observations,
  then stack the means. Positive and negative values accumulate separately
  from zero. Entirely missing components contribute no height, and groups with
  no observed components are omitted. Unused categorical levels are omitted.
- Newick strings are parsed without filesystem length checks. A string ending
  in `;` is treated as Newick; use `Path` for a filename that itself ends in `;`.
  File-read errors retain a specific diagnostic.
- Tree transfers and gene/species mapping copy the tree without recursion
  through parent/child links. Metadata is deeply copied with shared references
  and references to copied nodes preserved. Arbitrarily nested user metadata
  retains normal Python deepcopy limitations; the process recursion limit is
  never changed. ETE's existing rerooting rules may relocate or remove the old
  root's properties when a new root is introduced.
- Ancestor lookups accept structural columns and overlapping target/return
  columns in both scalar and prepared forms. Duplicate branch rows retain the
  existing first-row policy. Prepared lookup data remains independent of the
  source frame.
- `compute_delta` keeps unrelated columns such as `parent_value`, as well as
  row order and index, without mutating its input.
- OU identifier columns are read as text, preserving numeric names, leading
  zeros, and names such as `NA`. Empty names with an assigned regime are still
  invalid; numeric columns retain their numeric/missing-value validation.
- Taxonomic parsing recognizes periods on known qualifiers and rank aliases,
  including `cf.`, `aff.`, `nr.`, `subsp.`, `ssp.`, `var.`, and `f.`. Displayed
  scientific names round-trip, while punctuation in strain values is retained.
- `nuc_freq2theta` requires exactly the four supported keys A/T/C/G. Extra keys
  are rejected instead of silently changing the normalization denominator.

Public array, dataframe, parser, path, and result annotations have been made
more specific. Tree copying, dataframe/ancestor operations, and regression
calculations have separate internal modules. Development snapshots, clean
environment commands, static consumer tests, and consolidated CI are described
in the [development guide](development.md).
