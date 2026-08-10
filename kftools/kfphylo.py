import copy
import logging
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import ete4
import numpy as np

from ._validation import validate_boolean_flag
from .kfspecies import parse_species_label

logger = logging.getLogger(__name__)


def _load_tree_or_value_error(tree_source, parser=1, argument_name="tree_source"):
    try:
        return load_phylo_tree(tree_source, parser=parser)
    except TypeError as exc:
        raise ValueError(f"{argument_name} must be a Newick string, path, or ete4.PhyloTree instance") from exc


def _read_newick_source(tree_source):
    if isinstance(tree_source, os.PathLike):
        try:
            tree_source = os.fspath(tree_source)
        except TypeError as exc:
            raise TypeError("tree_source must be a Newick string, path, or ete4.PhyloTree instance") from exc
    if not isinstance(tree_source, str):
        raise TypeError("tree_source must be a Newick string, path, or ete4.PhyloTree instance")
    if tree_source.strip() == "":
        raise ValueError("tree_source must not be an empty string")
    tree_path = Path(tree_source)
    if not tree_path.exists():
        return tree_source, None
    if not tree_path.is_file():
        raise ValueError(f"Tree path exists but is not a file: {tree_path}")
    try:
        newick = tree_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read tree file: {tree_path}") from exc
    if newick.strip() == "":
        raise ValueError(f"Tree file is empty: {tree_path}")
    return newick, tree_path


def load_phylo_tree(tree_source: Any, parser: int = 1) -> Any:
    """Load a Newick string or path as an ETE tree, or return a supplied tree."""
    if isinstance(tree_source, ete4.PhyloTree):
        return tree_source
    if tree_source is None:
        raise ValueError("tree_source must not be None")
    newick, tree_path = _read_newick_source(tree_source)
    try:
        return ete4.PhyloTree(newick, parser=parser)
    except Exception as exc:
        if tree_path is not None:
            raise ValueError(f"Failed to parse tree file as Newick: {tree_path}") from exc
        raise ValueError("tree_source is neither a readable tree file path nor a valid Newick string") from exc


def get_tree_height(tree_file: Any) -> float:
    """Return the maximum root-to-tip branch length in a phylogenetic tree."""
    tree = _load_tree_or_value_error(tree_file, parser=1, argument_name="tree_file")
    leaves = list(tree.leaves())
    if len(leaves) == 0:
        return 0.0
    max_root_to_tip_distance = 0.0
    stack = [(tree, 0.0)]
    while stack:
        node, distance_from_root = stack.pop()
        if node.is_leaf:
            if distance_from_root > max_root_to_tip_distance:
                max_root_to_tip_distance = distance_from_root
            continue
        for child in node.get_children():
            child_dist = child.dist
            if isinstance(child_dist, bool) or (not isinstance(child_dist, (int, float, np.integer, np.floating))):
                raise ValueError("Tree branch lengths must be finite numeric values")
            child_dist = float(child_dist)
            if not np.isfinite(child_dist):
                raise ValueError("Tree branch lengths must be finite numeric values")
            if child_dist < 0:
                raise ValueError("Tree branch lengths must be non-negative")
            stack.append((child, distance_from_root + child_dist))
    return max_root_to_tip_distance


def _descendant_leafsets(tree):
    leafsets = {}
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf:
            leafsets[node] = frozenset((node.name,))
        else:
            leafsets[node] = frozenset().union(*(leafsets[child] for child in node.children))
    return leafsets


def _display_clade_signatures(signatures):
    return [sorted(signature) for signature in sorted(signatures, key=lambda value: (len(value), sorted(value)))]


def _internal_nodes_by_clade(tree, leafsets):
    nodes_by_clade: dict[frozenset[str], list[Any]] = {}
    for node in tree.traverse(strategy="postorder"):
        if not node.is_leaf:
            nodes_by_clade.setdefault(leafsets[node], []).append(node)
    return nodes_by_clade


def transfer_internal_node_names(tree_to: Any, tree_from: Any) -> Any:
    """Return a copy of ``tree_to`` with matching clade names from ``tree_from``.

    Both inputs are left unchanged and must contain identical leaf sets and
    rooted clade signatures.  Arbitrary internal node degrees are supported.
    """
    tree_to = copy.deepcopy(_load_tree_or_value_error(tree_to, parser=1, argument_name="tree_to"))
    tree_from = copy.deepcopy(_load_tree_or_value_error(tree_from, parser=1, argument_name="tree_from"))
    tree_to = add_numerical_node_labels(tree_to)
    tree_from = add_numerical_node_labels(tree_from)
    to_leafsets = _descendant_leafsets(tree_to)
    from_leafsets = _descendant_leafsets(tree_from)
    to_by_clade = _internal_nodes_by_clade(tree_to, to_leafsets)
    from_by_clade = _internal_nodes_by_clade(tree_from, from_leafsets)
    missing_clades = set(from_by_clade) - set(to_by_clade)
    extra_clades = set(to_by_clade) - set(from_by_clade)
    mismatch_clades = sorted(
        (
            clade
            for clade in set(to_by_clade) & set(from_by_clade)
            if len(to_by_clade[clade]) != len(from_by_clade[clade])
        ),
        key=lambda clade: (len(clade), sorted(clade)),
    )
    multiplicity_mismatches = [
        {
            "clade": sorted(clade),
            "tree_to": len(to_by_clade[clade]),
            "tree_from": len(from_by_clade[clade]),
        }
        for clade in mismatch_clades
    ]
    if missing_clades or extra_clades or multiplicity_mismatches:
        raise ValueError(
            "tree topologies are different; "
            f"missing_in_tree_to={_display_clade_signatures(missing_clades)}, "
            f"extra_in_tree_to={_display_clade_signatures(extra_clades)}, "
            f"clade_multiplicity_mismatches={multiplicity_mismatches}"
        )
    for clade, to_nodes in to_by_clade.items():
        for node_to, node_from in zip(to_nodes, from_by_clade[clade], strict=True):
            if node_from.name is not None:
                node_to.name = node_from.name
    return tree_to


def fill_internal_node_names(tree: Any) -> Any:
    """Assign deterministic descendant-based names to unnamed internal nodes."""
    tree = _load_tree_or_value_error(tree, parser=1, argument_name="tree")
    used_names = {node.name for node in tree.traverse() if isinstance(node.name, str) and node.name.strip() != ""}
    counter = 1
    for node in tree.traverse():
        node_name = node.name
        has_missing_name = (node_name is None) or (isinstance(node_name, str) and (node_name.strip() == ""))
        if (not node.is_leaf) and has_missing_name:
            candidate_name = f"n{counter}"
            while candidate_name in used_names:
                counter += 1
                candidate_name = f"n{counter}"
            node.name = candidate_name
            used_names.add(candidate_name)
            counter += 1
    return tree


def add_numerical_node_labels(tree: Any) -> Any:
    """Assign deterministic `branch_id` values in a CSUBST-compatible manner.

    The ranking algorithm intentionally mirrors CSUBST's branch-ID assignment
    so that identical tree topologies receive identical `branch_id` values
    across kftools and CSUBST.
    """
    tree = _load_tree_or_value_error(tree, parser=1, argument_name="tree")
    all_leaf_names = list(tree.leaf_names())
    invalid_leaf_names = [
        leaf_name for leaf_name in all_leaf_names if (not isinstance(leaf_name, str)) or (leaf_name.strip() == "")
    ]
    if len(invalid_leaf_names) > 0:
        raise ValueError("Tree leaf names must be non-empty strings for CSUBST-compatible branch_id assignment")
    leaf_name_counts = Counter(all_leaf_names)
    duplicate_leaf_names = sorted([leaf_name for leaf_name, count in leaf_name_counts.items() if count > 1])
    if len(duplicate_leaf_names) > 0:
        raise ValueError(
            f"Tree leaf names must be unique for CSUBST-compatible branch_id assignment: {duplicate_leaf_names}"
        )
    all_leaf_names = sorted(all_leaf_names)
    leaf_branch_ids = {leaf_name: (1 << i) for i, leaf_name in enumerate(all_leaf_names)}
    nodes = list(tree.traverse())
    signature_by_node = {}
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf:
            signature_by_node[node] = leaf_branch_ids[node.name]
        else:
            signature_by_node[node] = sum(signature_by_node[child] for child in node.children)
    clade_signatures = [signature_by_node[node] for node in nodes]
    sorted_node_indices = sorted(range(len(nodes)), key=lambda idx: clade_signatures[idx])
    rank_by_node_index = {node_index: rank for rank, node_index in enumerate(sorted_node_indices)}
    for node_index, node in enumerate(nodes):
        node.branch_id = rank_by_node_index[node_index]
    return tree


def _validate_transfer_root_leaf_names(leaf_names, tree_name):
    invalid_leaf_names = [
        leaf_name for leaf_name in leaf_names if (not isinstance(leaf_name, str)) or (leaf_name.strip() == "")
    ]
    if len(invalid_leaf_names) > 0:
        raise ValueError(f"{tree_name} leaf names must be non-empty strings for transfer_root")
    leaf_name_counts = Counter(leaf_names)
    duplicate_leaf_names = sorted([leaf_name for leaf_name, count in leaf_name_counts.items() if count > 1])
    if len(duplicate_leaf_names) > 0:
        raise ValueError(f"{tree_name} leaf names must be unique for transfer_root: {duplicate_leaf_names}")


def _resolve_clade_node(tree, clade_leafset):
    clade_leaf_names = sorted(clade_leafset)
    if len(clade_leaf_names) == 1:
        return next((leaf for leaf in tree.leaves() if leaf.name == clade_leaf_names[0]), None)
    try:
        clade_node = tree.common_ancestor(clade_leaf_names)
    except Exception:
        return None
    return clade_node if set(clade_node.leaf_names()) == clade_leafset else None


def _validate_identical_tip_sets(tree_to, tree_from):
    leaf_names_to_list = list(tree_to.leaf_names())
    leaf_names_from_list = list(tree_from.leaf_names())
    _validate_transfer_root_leaf_names(leaf_names_to_list, "tree_to")
    _validate_transfer_root_leaf_names(leaf_names_from_list, "tree_from")
    leaf_names_to = set(leaf_names_to_list)
    leaf_names_from = set(leaf_names_from_list)
    if leaf_names_to != leaf_names_from:
        raise ValueError(
            "tree_to and tree_from must have identical tips. "
            f"missing_in_tree_to={sorted(leaf_names_from - leaf_names_to)}, "
            f"extra_in_tree_to={sorted(leaf_names_to - leaf_names_from)}"
        )


def _choose_outgroup_split(tree_to, split_leafsets):
    valid_indices = [
        idx for idx, leafset in enumerate(split_leafsets) if _resolve_clade_node(tree_to, leafset) is not None
    ]
    if len(valid_indices) == 0:
        split_display = [sorted(leafset) for leafset in split_leafsets]
        raise ValueError(
            f"Failed to transfer root because tree_to does not contain the root split from tree_from: {split_display}"
        )
    if len(valid_indices) == 1:
        return valid_indices[0]
    return 0 if len(split_leafsets[0]) <= len(split_leafsets[1]) else 1


def _root_partition(tree, leafsets=None):
    leafsets = _descendant_leafsets(tree) if leafsets is None else leafsets
    return frozenset(leafsets[child] for child in tree.children)


def _incident_partition(node, leafsets, all_leaves):
    components = [leafsets[child] for child in node.children]
    if not node.is_root:
        parent_side = all_leaves - leafsets[node]
        if parent_side:
            components.append(parent_side)
    return frozenset(components)


def _find_root_vertex(tree, expected_partition):
    leafsets = _descendant_leafsets(tree)
    all_leaves = leafsets[tree]
    candidates = [
        node for node in tree.traverse() if _incident_partition(node, leafsets, all_leaves) == expected_partition
    ]
    if len(candidates) != 1:
        partition_display = _display_clade_signatures(expected_partition)
        raise ValueError(
            "Failed to transfer multifurcating root because tree_to does not contain a unique vertex "
            f"with root partition: {partition_display}; candidates={len(candidates)}"
        )
    return candidates[0]


def _validated_branch_distance(distance, error_prefix, allow_none=False):
    if (distance is None) and allow_none:
        return 0.0
    numeric_description = "finite numeric values" if error_prefix.endswith("lengths") else "a finite numeric value"
    if isinstance(distance, bool) or (not isinstance(distance, (int, float, np.integer, np.floating))):
        raise ValueError(f"{error_prefix} must be {numeric_description}")
    distance = float(distance)
    if not np.isfinite(distance):
        raise ValueError(f"{error_prefix} must be {numeric_description}")
    if distance < 0:
        raise ValueError(f"{error_prefix} must be non-negative")
    return distance


def _reroot_to_split(tree_to, split_leafsets, outgroup_idx, verbose):
    outgroups = sorted(split_leafsets[outgroup_idx])
    ingroups = sorted(split_leafsets[1 - outgroup_idx])
    root_dist = _validated_branch_distance(tree_to.dist, "tree_to root branch length", allow_none=True)
    if root_dist != 0.0:
        tree_to.dist = 0.0
    if verbose:
        logger.info("outgroups: %s", outgroups)
    tree_to.set_outgroup(ingroups[0])
    outgroup_ancestor = _resolve_clade_node(tree_to, set(outgroups))
    if outgroup_ancestor is None:
        raise ValueError(
            "Failed to transfer root because tree_to does not preserve the outgroup clade after rerooting."
        )
    tree_to.set_outgroup(outgroup_ancestor)


def _reroot_to_vertex(tree_to, expected_partition, verbose):
    root_dist = _validated_branch_distance(tree_to.dist, "tree_to root branch length", allow_none=True)
    if root_dist != 0.0:
        tree_to.dist = 0.0
    root_vertex = _find_root_vertex(tree_to, expected_partition)
    if verbose:
        logger.info("multifurcating root partition: %s", _display_clade_signatures(expected_partition))
    if not root_vertex.is_root:
        try:
            tree_to.set_outgroup(root_vertex)
            tree_to.unroot()
        except (AssertionError, ValueError) as exc:
            raise ValueError("Failed to reroot tree_to at the multifurcating root vertex") from exc
    if _root_partition(tree_to) != expected_partition:
        raise ValueError("Failed to preserve the requested multifurcating root partition in tree_to")


def _validated_subroot_distances(nodes, tree_name):
    return [_validated_branch_distance(node.dist, f"{tree_name} root child branch lengths") for node in nodes]


def _transfer_subroot_distances(subroot_to, subroot_from):
    distances_to = _validated_subroot_distances(subroot_to, "tree_to")
    distances_from = _validated_subroot_distances(subroot_from, "tree_from")
    total_to = sum(distances_to)
    total_from = sum(distances_from)
    if total_from <= 0:
        return
    dist_by_leafset = {
        frozenset(node.leaf_names()): distance for node, distance in zip(subroot_from, distances_from, strict=True)
    }
    for node_to in subroot_to:
        node_from_dist = dist_by_leafset.get(frozenset(node_to.leaf_names()))
        if node_from_dist is None:
            raise ValueError(
                "Failed to transfer root because rerooted split in tree_to did not match tree_from root split."
            )
        node_to.dist = (node_from_dist / total_from) * total_to


def transfer_root(tree_to: Any, tree_from: Any, verbose: bool = False) -> Any:
    """Return a rerooted copy of ``tree_to`` using the root in ``tree_from``.

    Work is performed on a deep copy so a validation or rerooting failure never
    leaves the caller's tree partially modified.  Binary roots are transferred
    onto a matching edge; roots with three or more children require a unique
    existing vertex with the same incident leaf partition.
    """
    verbose = validate_boolean_flag(verbose, "verbose")
    tree_to = copy.deepcopy(_load_tree_or_value_error(tree_to, parser=1, argument_name="tree_to"))
    tree_from = _load_tree_or_value_error(tree_from, parser=1, argument_name="tree_from")
    _validate_identical_tip_sets(tree_to, tree_from)

    from_children = tree_from.get_children()
    if len(from_children) < 2:
        raise ValueError(f"tree_from root must contain at least 2 children, got {len(from_children)}")

    split_leafsets = [frozenset(node.leaf_names()) for node in from_children]
    if len(from_children) == 2:
        outgroup_idx = _choose_outgroup_split(tree_to, split_leafsets)
        _reroot_to_split(tree_to, split_leafsets, outgroup_idx, verbose)
    else:
        _reroot_to_vertex(tree_to, frozenset(split_leafsets), verbose)

    subroot_to = tree_to.get_children()
    if len(subroot_to) != len(from_children):
        raise ValueError(
            "Failed to transfer root because rerooted tree_to root degree does not match tree_from "
            f"(expected {len(from_children)}, got {len(subroot_to)} children)."
        )
    if len(from_children) == 2:
        _transfer_subroot_distances(subroot_to, from_children)
    else:
        _validated_subroot_distances(subroot_to, "tree_to")
        _validated_subroot_distances(from_children, "tree_from")

    for n_to in tree_to.traverse():
        if not n_to.name:
            n_to.name = tree_to.name or "Root"
            tree_to.name = "Root"
            break
    return tree_to


def _validate_ultrametric_tolerance(tol):
    if isinstance(tol, bool) or (not isinstance(tol, (int, float, np.integer, np.floating))):
        raise ValueError("tol must be a finite numeric value")
    if not np.isfinite(float(tol)):
        raise ValueError("tol must be a finite numeric value")
    tol = float(tol)
    if tol < 0:
        raise ValueError("tol must be non-negative")
    return tol


def _root_to_tip_extrema(tree):
    min_dist = np.inf
    max_dist = -np.inf
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
                child_dist = _validated_branch_distance(child.dist, "All branch lengths")
                stack.append((child, distance_from_root + child_dist))
    if np.isinf(min_dist):
        min_dist = 0.0
        max_dist = 0.0
        min_dist_leaf = tree.name
        max_dist_leaf = tree.name
    return min_dist, max_dist, min_dist_leaf, max_dist_leaf


def check_ultrametric(tree: Any, tol: float = 0, verbose: bool = False) -> bool:
    """Return whether root-to-tip distances agree within an absolute tolerance."""
    tree = _load_tree_or_value_error(tree, parser=1, argument_name="tree")
    tol = _validate_ultrametric_tolerance(tol)
    verbose = validate_boolean_flag(verbose, "verbose")
    min_dist, max_dist, min_dist_leaf, max_dist_leaf = _root_to_tip_extrema(tree)

    dif_tree_length = max_dist - min_dist
    is_ultrametric = dif_tree_length <= tol
    if (dif_tree_length > tol) and verbose:
        logger.warning(
            "(max - min) root-to-tip path (%s) was bigger than tol (%s); min=%s in %s, max=%s in %s",
            dif_tree_length,
            tol,
            min_dist,
            min_dist_leaf,
            max_dist,
            max_dist_leaf,
        )
    return is_ultrametric


def _parse_taxonomic_leaves(tree, species_parser):
    leaves = list(tree.leaves())
    taxonomy_queries = []
    for leaf in leaves:
        leaf_name = leaf.name
        if (not isinstance(leaf_name, str)) or (leaf_name.strip() == ""):
            raise ValueError(
                f"Leaf name must be a non-empty string containing genus and species separated by '_': {leaf_name}"
            )
        try:
            parsed_species = parse_species_label(leaf_name, species_parser=species_parser)
        except ValueError as exc:
            raise ValueError(f"Leaf name must contain genus and species separated by '_': {leaf_name}") from exc
        leaf.sci_name = parsed_species.scientific_name
        leaf.taxonomy_query = parsed_species.taxonomy_query
        taxonomy_queries.append(leaf.taxonomy_query)
    return leaves, taxonomy_queries


def _load_ncbi_taxa(taxonomy_queries):
    try:
        ncbi = ete4.NCBITaxa()
    except Exception as exc:
        raise ValueError("Failed to initialize NCBITaxa database") from exc
    try:
        name2id = ncbi.get_name_translator(names=list(set(taxonomy_queries)))
    except Exception as exc:
        raise ValueError("Failed to query scientific names in NCBITaxa") from exc
    return ncbi, name2id


def _assign_taxids(leaves, name2id):
    for leaf in leaves:
        taxids = name2id.get(leaf.taxonomy_query, [])
        if len(taxids) == 0:
            raise ValueError(f"No taxid found for scientific name: {leaf.taxonomy_query}")
        if len(taxids) > 1:
            warnings.warn(
                f"{leaf.taxonomy_query} has {len(taxids)} taxids; using the first entry.",
                RuntimeWarning,
                stacklevel=2,
            )
        leaf.taxid = taxids[0]


def taxonomic_annotation(tree: Any, species_parser: Any = None, parser: Any = None) -> Any:
    """Annotate a tree with ETE NCBI taxonomy using parsed leaf species labels."""
    if parser is not None:
        if species_parser is not None:
            raise ValueError("Use only one of species_parser or parser")
        species_parser = parser
    tree = _load_tree_or_value_error(tree, parser=1, argument_name="tree")
    leaves, taxonomy_queries = _parse_taxonomic_leaves(tree, species_parser)
    ncbi, name2id = _load_ncbi_taxa(taxonomy_queries)
    _assign_taxids(leaves, name2id)
    try:
        ncbi.annotate_tree(tree, taxid_attr="taxid")
    except Exception as exc:
        raise ValueError("Failed to annotate tree with NCBI taxonomy") from exc
    return tree
