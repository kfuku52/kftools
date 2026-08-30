"""Internal tree copying shared by phylogenetic and orthogroup operations."""

import copy

from ete4 import PhyloTree


def copy_tree(tree: PhyloTree) -> PhyloTree:
    """Deep-copy a subtree without recursing through its parent/child links.

    Build the topology first, then use one deepcopy memo for metadata. This
    preserves shared attributes and references to nodes within the copied tree,
    including cyclic references, without a recursion limit proportional to tree
    depth. Arbitrarily nested user metadata follows normal deepcopy semantics.
    """
    nodes = list(tree.traverse())
    memo = {id(node): copy.copy(node) for node in nodes}
    for node in nodes:
        clone = memo[id(node)]
        clone.children = [memo[id(child)] for child in node.children]
        clone.up = memo.get(id(node.up))
        memo[id(node.children)] = clone.children
    for node in nodes:
        clone = memo[id(node)]
        clone.props = copy.deepcopy(node.props, memo)
        clone.__dict__ = copy.deepcopy(node.__dict__, memo)
    return memo[id(tree)]
