import os
import sys
from pathlib import Path

import ete4
import numpy


def load_phylo_tree(tree_source, parser=1):
    if isinstance(tree_source, ete4.PhyloTree):
        return tree_source
    if tree_source is None:
        raise ValueError("tree_source must not be None")
    if isinstance(tree_source, os.PathLike):
        tree_source = os.fspath(tree_source)
    if isinstance(tree_source, str):
        if tree_source.strip() == "":
            raise ValueError("tree_source must not be an empty string")
        tree_path = Path(tree_source)
        if tree_path.exists():
            if not tree_path.is_file():
                raise ValueError(f"Tree path exists but is not a file: {tree_path}")
            try:
                with tree_path.open() as f:
                    tree_source = f.read()
            except OSError as exc:
                raise ValueError(f"Failed to read tree file: {tree_path}") from exc
            if tree_source.strip() == "":
                raise ValueError(f"Tree file is empty: {tree_path}")
        return ete4.PhyloTree(tree_source, parser=parser)
    raise TypeError("tree_source must be a Newick string, path, or ete4.PhyloTree instance")


def get_tree_height(tree_file):
    tree = load_phylo_tree(tree_file, parser=1)
    return tree.get_distance(target=next(tree.leaves()))


def transfer_internal_node_names(tree_to, tree_from):
    tree_to = load_phylo_tree(tree_to, parser=1)
    tree_from = load_phylo_tree(tree_from, parser=1)
    rf_dist = tree_to.robinson_foulds(tree_from)[0]
    if rf_dist != 0:
        raise ValueError('tree topologies are different. RF distance = ' + str(rf_dist))
    tree_to = add_numerical_node_labels(tree_to)
    tree_from = add_numerical_node_labels(tree_from)
    name_by_label = {}
    for node in tree_from.traverse():
        if not node.is_leaf:
            name_by_label[node.branch_id] = node.name
    for node in tree_to.traverse():
        if node.is_leaf:
            continue
        matched_name = name_by_label.get(node.branch_id)
        if matched_name is not None:
            node.name = matched_name
    return tree_to


def fill_internal_node_names(tree):
    counter = 1
    for node in tree.traverse():
        if (not node.is_leaf) and (node.name == ''):
            node.name = 'n' + str(counter)
            counter += 1
    return tree


def add_numerical_node_labels(tree):
    """Assign deterministic `branch_id` values in a CSUBST-compatible manner.

    The ranking algorithm intentionally mirrors CSUBST's branch-ID assignment
    so that identical tree topologies receive identical `branch_id` values
    across kftools and CSUBST.
    """
    all_leaf_names = sorted(tree.leaf_names())
    leaf_branch_ids = {leaf_name: (1 << i) for i, leaf_name in enumerate(all_leaf_names)}
    nodes = list(tree.traverse())
    clade_signatures = []
    for node in nodes:
        leaf_names = node.leaf_names()
        clade_signature = sum(leaf_branch_ids[leaf_name] for leaf_name in leaf_names)
        clade_signatures.append(clade_signature)
    sorted_node_indices = sorted(range(len(nodes)), key=lambda idx: clade_signatures[idx])
    rank_by_node_index = {node_index: rank for rank, node_index in enumerate(sorted_node_indices)}
    for node_index, node in enumerate(nodes):
        node.branch_id = rank_by_node_index[node_index]
    return tree


def transfer_root(tree_to, tree_from, verbose=False):
    tree_to = load_phylo_tree(tree_to, parser=1)
    tree_from = load_phylo_tree(tree_from, parser=1)
    leaf_names_to = set(tree_to.leaf_names())
    leaf_names_from = set(tree_from.leaf_names())
    if leaf_names_to != leaf_names_from:
        missing_in_tree_to = sorted(leaf_names_from - leaf_names_to)
        extra_in_tree_to = sorted(leaf_names_to - leaf_names_from)
        raise ValueError(
            "tree_to and tree_from must have identical tips. "
            f"missing_in_tree_to={missing_in_tree_to}, extra_in_tree_to={extra_in_tree_to}"
        )

    from_children = tree_from.get_children()
    if len(from_children) != 2:
        raise ValueError(f"tree_from root must be bifurcating (2 children), got {len(from_children)}")

    split_leafsets = [set(node.leaf_names()) for node in from_children]

    def _resolve_clade_node(tree, clade_leafset):
        clade_leaf_names = sorted(clade_leafset)
        if len(clade_leaf_names) == 1:
            leaf_name = clade_leaf_names[0]
            for leaf in tree.leaves():
                if leaf.name == leaf_name:
                    return leaf
            return None
        try:
            clade_node = tree.common_ancestor(clade_leaf_names)
        except Exception:
            return None
        if set(clade_node.leaf_names()) != clade_leafset:
            return None
        return clade_node

    valid_outgroup_indices = [
        idx for idx, clade_leafset in enumerate(split_leafsets)
        if _resolve_clade_node(tree_to, clade_leafset) is not None
    ]
    if len(valid_outgroup_indices) == 0:
        split_display = [sorted(clade_leafset) for clade_leafset in split_leafsets]
        raise ValueError(
            "Failed to transfer root because tree_to does not contain "
            f"the root split from tree_from: {split_display}"
        )
    if len(valid_outgroup_indices) == 1:
        outgroup_idx = valid_outgroup_indices[0]
    else:
        outgroup_idx = 0 if len(split_leafsets[0]) <= len(split_leafsets[1]) else 1
    ingroup_idx = 1 - outgroup_idx

    outgroups = sorted(split_leafsets[outgroup_idx])
    ingroups = sorted(split_leafsets[ingroup_idx])
    if verbose:
        print('outgroups:', outgroups)
    tree_to.set_outgroup(ingroups[0])

    outgroup_ancestor = _resolve_clade_node(tree_to, set(outgroups))
    if outgroup_ancestor is None:
        raise ValueError(
            "Failed to transfer root because tree_to does not preserve "
            "the outgroup clade after rerooting."
        )
    tree_to.set_outgroup(outgroup_ancestor)

    subroot_to = tree_to.get_children()
    if len(subroot_to) != 2:
        raise ValueError(
            "Failed to transfer root because rerooted tree_to root is not bifurcating "
            f"(got {len(subroot_to)} children)."
        )
    subroot_from = from_children
    total_subroot_length_to = sum(node.dist for node in subroot_to)
    total_subroot_length_from = sum(node.dist for node in subroot_from)
    if total_subroot_length_from > 0:
        dist_by_leafset = {
            frozenset(node.leaf_names()): node.dist
            for node in subroot_from
        }
        for n_to in subroot_to:
            n_to_leafset = frozenset(n_to.leaf_names())
            n_from_dist = dist_by_leafset.get(n_to_leafset)
            if n_from_dist is None:
                raise ValueError(
                    "Failed to transfer root because rerooted split in tree_to "
                    "did not match tree_from root split."
                )
            n_to.dist = (n_from_dist / total_subroot_length_from) * total_subroot_length_to

    for n_to in tree_to.traverse():
        if not n_to.name:
            n_to.name = tree_to.name or 'Root'
            tree_to.name = 'Root'
            break
    return tree_to


def check_ultrametric(tree, tol=0):
    min_dist = numpy.inf
    max_dist = -numpy.inf
    min_dist_leaf = None
    max_dist_leaf = None
    stack = [(tree, 0.0)]
    while stack:
        node, distance_from_root = stack.pop()
        if node.is_leaf:
            if distance_from_root < min_dist:
                min_dist = distance_from_root
                min_dist_leaf = node.name
            if distance_from_root > max_dist:
                max_dist = distance_from_root
                max_dist_leaf = node.name
        else:
            for child in node.get_children():
                stack.append((child, distance_from_root + child.dist))

    if numpy.isinf(min_dist):
        min_dist = 0.0
        max_dist = 0.0
        min_dist_leaf = tree.name
        max_dist_leaf = tree.name

    if tol == 0:
        tol = max_dist * 0.001
    dif_tree_length = max_dist - min_dist
    is_ultrametric = dif_tree_length < tol
    if dif_tree_length > tol:
        sys.stderr.write('(max - min) root-to-tip path ({}) was bigger than tol ({}).\n'.format(dif_tree_length, tol))
        sys.stderr.write('min_dist_leaf = {:,} in {}\n'.format(min_dist, min_dist_leaf))
        sys.stderr.write('max_dist_leaf = {:,} in {}\n'.format(max_dist, max_dist_leaf))
    return is_ultrametric


def taxonomic_annotation(tree):
    leaves = list(tree.leaves())
    sci_names = []
    for leaf in leaves:
        leaf_name_split = leaf.name.split("_")
        if len(leaf_name_split) < 2:
            raise ValueError(f"Leaf name must contain genus and species separated by '_': {leaf.name}")
        binom_name = leaf_name_split[0] + " " + leaf_name_split[1]
        leaf.sci_name = binom_name
        sci_names.append(leaf.sci_name)
    ncbi = ete4.NCBITaxa()
    name2id = ncbi.get_name_translator(names=list(set(sci_names)))
    for leaf in leaves:
        taxids = name2id.get(leaf.sci_name, [])
        if len(taxids) == 0:
            raise ValueError(f"No taxid found for scientific name: {leaf.sci_name}")
        if len(taxids) > 1:
            print(leaf.sci_name, "has", len(taxids), "taxids.")
        leaf.taxid = taxids[0]
    ncbi.annotate_tree(tree, taxid_attr="taxid")
    return tree
