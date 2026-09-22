# Changes in 0.6.6

- Preserve exact integer differences in `compute_delta`, including nullable
  unsigned values and differences outside int64 bounds. Integer deltas use an
  object column; missing values remain missing.
- Add nullable species branch IDs to `node_gene2species` so unnamed and repeated
  ancestor names are distinguishable without changing the legacy name columns.
- Reject OU trait names that collide with derived statistics and ambiguous
  duplicate tissue columns in `calc_tau`.
- Parse regime IDs exactly with missing rows and compute max-ID-plus-one using
  Python integers, preserving values above `2**53` and at the int64 limit.
- Honor string colors and general category-to-color mappings in `hist_boxplot`.
- Normalize inverse branch weights to avoid overflow for tiny positive lengths.
- Reject different single-tip trees in `transfer_internal_node_names`.

See [data semantics](data-semantics.md) and [file formats](file-formats.md) for
the output and validation contracts.
