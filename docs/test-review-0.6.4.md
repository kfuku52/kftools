# Test review for 0.6.4

The review covered every `tests/test_*.py` file, the static consumer checks in
`tests/typing/public_api.py`, and `scripts/wheel_smoke.py`. The decision for each
case was which realistic defect it detects relative to setup, fragility, and
maintenance cost. Test counts and coverage percentages were not preservation
targets.

| Area | Removed or consolidated | Defects still checked |
| --- | --- | --- |
| Public API metadata | Deleted reflective docstring/annotation-presence tests. | Consumer type checks still detect incorrect public return types and accidental `Any`; wheel checks verify the shipped `py.typed` marker. |
| Expression | Removed shape-only tau smoke calls and warning-message inspection. | Exact tau/complementarity results, zero rows, negative/nonfinite data and exponent overflow; unexpected warnings fail via pytest. |
| Sequences | Removed repeated malformed path/model types, unknown-base value permutations, duplicate normalization and zero-length binary weighting cases. | Frequency normalization, ambiguous bases, FASTA identity/length problems, all-child weighting, zero lengths, order invariance and invalid parameters. |
| Statistics | Removed unseeded random input with only a tuple-length assertion. | Numerical comparison of the custom Brunner–Munzel implementation with SciPy, tail direction, filtering and undefined variance. |
| Tree labels | Replaced a copied bit-signature ranking algorithm with explicit small-tree IDs and an analytically numbered 65-leaf star. | CSUBST ordering, child-order independence, integer width and ambiguous labels. |
| Tree transfer/copy | Removed a direct private-copy test and repetitive invalid wrapper types; moved root-reference checking into public transfers. | Topology, branch distances, deep trees, input preservation, shared/cyclic metadata and both transfer operations. |
| Ancestor lookup | Removed internal table identity checks and same-implementation comparison. | Explicit results across orthogroups, overlapping column roles, cycles, missing parents and input preservation. |
| OU/regime tables | Merged shuffled-column success tests; shared file fixtures; removed repeated regime validation through both APIs and file-type matrices. | Conflicting/unknown node mappings, duplicate names, missing traits, fractional/negative/overflowing IDs, numerical outputs and multifurcating relationships. |
| Plotting | Removed random smoke plots, most unsupported-Python-type cases, repeated category validation and redundant warning-string checks. | Stacking values/signs, category loss/merging, colors, regression values, row alignment, constant/two-point inputs, log-link zeros, invalid ranges and figure cleanup. |
| Species/utilities | Removed unrelated tree/sequence smoke checks from species tests and repeated channel-validation cases. | Scientific-name round trips, parser configurations, taxonomy queries, RGB conversion and gradient endpoints. |
| Regression matrices | Reduced equivalent missing-token spellings, frequency cardinalities, predictor names and orientation/dtype products. | Leading-zero and missing-like identifiers, model-specific cardinality, reserved/non-identifier names, both orientations and nullable values. |

External-boundary fakes remain where they detect our behavior: NCBI failure
translation and query construction, unreadable tree paths, and preventing FIFO
reads. They do not merely assert a mock's configured return value. The installed
wheel smoke test remains because source-tree tests cannot detect missing package
files or incorrect installed metadata.

The suite deliberately no longer promises an exact error for every unsupported
Python object or option spelling, blanket documentation presence, or private
cache identity. It retains the existing warning policy and coverage gate without
adding filler cases to recover removed coverage. No runtime library behavior or
public format changed.
