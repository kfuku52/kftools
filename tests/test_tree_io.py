"""Tree input and non-destructive copying across sizes and metadata shapes."""

import os
import sys
from pathlib import Path

import ete4
import numpy as np
import pytest

from kftools import kfog, kfphylo
from kftools._tree import copy_tree


def comb_tree(leaves):
    value = "(S0_x:1,S1_x:1)N1:1"
    for i in range(2, leaves):
        value = f"({value},S{i}_x:1)N{i}:1"
    return ete4.PhyloTree(value + ";", parser=1)


def test_long_newick_is_not_sent_to_the_filesystem(monkeypatch):
    newick = "(" + ",".join(f"leaf{i}:1" for i in range(1000)) + ");"

    def unexpected_read(*args, **kwargs):
        pytest.fail("Newick input must not be treated as a path")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    loaded = kfphylo.load_phylo_tree(newick)
    assert list(loaded.leaf_names()) == list(ete4.PhyloTree(newick, parser=1).leaf_names())


def test_tree_paths_remain_unambiguous_and_report_read_errors(tmp_path, monkeypatch):
    tree_file = tmp_path / "(a,b);"
    tree_file.write_text("(A:1,B:2)Root;")
    assert list(kfphylo.load_phylo_tree(tree_file).leaf_names()) == ["A", "B"]
    with pytest.raises(ValueError, match="Failed to read tree file"):
        kfphylo.load_phylo_tree(tmp_path / "missing.nwk")

    def unreadable(*args, **kwargs):
        raise PermissionError("no access")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(ValueError, match="Failed to read tree file"):
        kfphylo.load_phylo_tree(tree_file)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires named pipes")
def test_non_regular_tree_path_is_rejected_without_opening(tmp_path, monkeypatch):
    pipe = tmp_path / "tree.pipe"
    os.mkfifo(pipe)

    def unexpected_open(*args, **kwargs):
        pytest.fail("a tree loader must not open a pipe and block")

    monkeypatch.setattr(Path, "read_text", unexpected_open)
    with pytest.raises(ValueError, match="not a file"):
        kfphylo.load_phylo_tree(pipe)


@pytest.mark.parametrize("operation", ["transfer_root", "transfer_internal_node_names"])
def test_deep_tree_transfer_preserves_inputs_and_recursion_limit(operation):
    tree = comb_tree(1000)
    before = tree.write(parser=1, format_root_node=True)
    limit = sys.getrecursionlimit()
    result = getattr(kfphylo, operation)(tree, tree)
    assert result is not tree
    assert len(list(result.leaves())) == 1000
    assert tree.write(parser=1, format_root_node=True) == before
    assert sys.getrecursionlimit() == limit
    assert all(child.up is node for node in result.traverse() for child in node.children)


@pytest.mark.parametrize("operation", ["transfer_root", "transfer_internal_node_names"])
def test_tree_transfer_deep_copies_shared_metadata_and_node_references(operation):
    tree = ete4.PhyloTree("((A:1,B:2)Inner:3,C:4)Root;", parser=1)
    leaf = next(tree.leaves())
    shared = [1, 2]
    tree.custom = {"leaf": leaf, "shared": shared, "children": tree.children}
    leaf.props["shared"] = shared
    leaf.props["self"] = leaf
    leaf.props["array"] = np.array([2, 3])
    result = getattr(kfphylo, operation)(tree, tree)
    result_leaf = next(n for n in result.leaves() if n.name == leaf.name)
    assert result.custom["leaf"] is result_leaf
    assert result.custom["shared"] is result_leaf.props["shared"]
    assert result_leaf.props["self"] is result_leaf
    result_leaf.props["shared"].append(3)
    result_leaf.props["array"][0] = 99
    assert shared == [1, 2]
    assert leaf.props["array"].tolist() == [2, 3]
    assert all(child.up is node for node in tree.traverse() for child in node.children)


def test_copy_preserves_root_metadata_before_ete_rerooting():
    tree = comb_tree(250)
    tree.props["self"] = tree
    tree.custom = {"children": tree.children, "props": tree.props}
    copied = copy_tree(tree)
    assert copied.props["self"] is copied
    assert copied.custom["children"] is copied.children
    assert copied.custom["props"] is copied.props
    assert copied.up is None


def test_gene_species_mapping_can_copy_deep_gene_trees():
    gene = comb_tree(250)
    species = ete4.PhyloTree("(" + ",".join(f"S{i}_x:1" for i in range(250)) + ")Root;", parser=1)
    before = gene.write(parser=1, format_root_node=True)
    result = kfog.node_gene2species(gene, species)
    assert result.shape[0] == 499
    assert result["branch_id"].is_unique
    assert gene.write(parser=1, format_root_node=True) == before
