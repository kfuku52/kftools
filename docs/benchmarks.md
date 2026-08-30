# Tree transfer measurements

Measured on macOS arm64, CPython 3.14.0, ETE4 4.4.0, on 2026-08-31.
Both revisions used the same interpreter/dependencies. Baseline: `838e74e`
(version 0.5.0); updated: version 0.6.0. Each row is a fresh process with five
calls; time is the median wall time in milliseconds. Peak RSS is the whole
process in MiB, including imports, input construction, and all five calls.

The source and target have the same topology/root. Existing transfer code still
performs its normal root/clade work; these numbers do not predict every possible
rerooting workload. Correctness tests separately exercise different root
placements, branch lengths, and polytomies.

| Operation | Shape / leaves | Before ms | After ms | Before peak MiB | After peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| transfer_internal_node_names | balanced / 1000 | 34.08 | 23.83 | 130.1 | 132.6 |
| transfer_internal_node_names | star / 1000 | 14.33 | 8.94 | 119.8 | 121.8 |
| transfer_internal_node_names | comb / 50 | 1.83 | 1.12 | 115.6 | 115.3 |
| transfer_internal_node_names | comb / 1000 | RecursionError | 69.11 | 117.2 | 204.3 |
| transfer_root | balanced / 1000 | 17.96 | 13.58 | 120.8 | 121.2 |
| transfer_root | star / 1000 | 16.23 | 12.53 | 112.5 | 118.4 |
| transfer_root | comb / 50 | 1.02 | 0.69 | 116.0 | 115.7 |
| transfer_root | comb / 1000 | RecursionError | 22.46 | 117.0 | 122.5 |

All six completed baseline cases had identical serialized output hashes after
the change, and input trees remained unchanged. Both 1,000-leaf comb cases now
complete; baseline memory for those rows describes a failed operation and is
not a like-for-like memory comparison. Completed cases showed small RSS changes
as well as lower measured times. The internal-name operation still constructs
clade sets, whose memory can grow quadratically on highly unbalanced trees;
removing recursive copying does not remove that separate scaling limit.

These are local measurements, not a guaranteed speedup. Reproduce them with
[`benchmarks/tree_transfers.py`](../benchmarks/tree_transfers.py); individual
samples and hashes are in
[`results-20260831.json`](../benchmarks/results-20260831.json).
