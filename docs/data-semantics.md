# Data and mutation semantics

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


See [changes in 0.6.0](changes-0.6.0.md) for corrected missing-value aggregation,
regression statistics, identifiers, and copying behavior.
