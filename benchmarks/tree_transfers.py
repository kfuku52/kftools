"""Measure tree transfers in an isolated process, optionally against old sources.

Example: python benchmarks/tree_transfers.py --shape comb --leaves 1000
Run each configuration in a new process for comparable peak RSS measurements.
"""

import argparse
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path


def newick(shape: str, leaves: int) -> str:
    nodes = [f"L{i}:1" for i in range(leaves)]
    if shape == "star":
        return f"({','.join(nodes)})Root:1;"
    if shape == "comb":
        value = nodes[0]
        for i, node in enumerate(nodes[1:]):
            value = f"({value},{node})N{i}:1"
        return value + ";"
    count = 0
    while len(nodes) > 1:
        paired = []
        for i in range(0, len(nodes), 2):
            group = nodes[i : i + 2]
            paired.append(f"({','.join(group)})N{count}:1" if len(group) == 2 else group[0])
            count += 1
        nodes = paired
    return nodes[0] + ";"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shape", choices=["balanced", "comb", "star"], default="balanced")
    parser.add_argument("--leaves", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--operation", choices=["transfer_root", "transfer_internal_node_names"], default="transfer_internal_node_names"
    )
    args = parser.parse_args()
    if args.leaves < 2 or args.repeat < 1:
        parser.error("--leaves must be >= 2 and --repeat must be >= 1")
    sys.path.insert(0, str(args.source.resolve()))
    import ete4

    from kftools import kfphylo

    if not Path(kfphylo.__file__).resolve().is_relative_to(args.source.resolve()):
        parser.error("--source must contain the kftools package being benchmarked")
    tree = ete4.PhyloTree(newick(args.shape, args.leaves), parser=1)
    original = tree.write(parser=1, format_root_node=True)
    operation = getattr(kfphylo, args.operation)
    durations = []
    digest = None
    error = None
    for _ in range(args.repeat):
        start = time.perf_counter()
        try:
            result = operation(tree, tree)
        except RecursionError:
            error = "RecursionError"
            break
        durations.append(time.perf_counter() - start)
        output = result.write(parser=1, format_root_node=True)
        current_digest = hashlib.sha256(output.encode()).hexdigest()
        assert digest in (None, current_digest), "non-deterministic output"
        digest = current_digest
        assert result is not tree
        assert tree.write(parser=1, format_root_node=True) == original, "input was mutated"
        del result
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        json.dumps(
            {
                "source": str(args.source),
                "python": sys.version.split()[0],
                "shape": args.shape,
                "leaves": args.leaves,
                "operation": args.operation,
                "median_seconds": statistics.median(durations) if durations else None,
                "seconds": durations,
                "peak_rss_mib": peak / (1024**2 if sys.platform == "darwin" else 1024),
                "sha256": digest,
                "error": error,
            }
        )
    )


if __name__ == "__main__":
    main()
