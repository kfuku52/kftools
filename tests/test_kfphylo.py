import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ete4
import matplotlib
import numpy as np

matplotlib.use("Agg")

from kftools import kfphylo


class TestKFPhylo(unittest.TestCase):
    def test_kfphylo(self):
        tree = ete4.PhyloTree("((A:1,B:1):2,C:3);", parser=1)
        out = kfphylo.add_numerical_node_labels(tree)
        labels = [node.branch_id for node in out.traverse()]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(kfphylo.check_ultrametric(tree))
        self.assertTrue(kfphylo.check_ultrametric("((A:1,B:1):2,C:3);"))
        self.assertAlmostEqual(kfphylo.get_tree_height("((A:1,B:1):2,C:3);"), 3.0)
        self.assertAlmostEqual(kfphylo.get_tree_height("(A:1,B:5);"), 5.0)
        with self.assertRaisesRegex(ValueError, "tree_file must be a Newick string"):
            kfphylo.get_tree_height(0)
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfphylo.get_tree_height("(A,B);")
        inf_height_tree = ete4.PhyloTree("(A:1,B:1);", parser=1)
        inf_height_tree.children[0].dist = np.inf
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfphylo.get_tree_height(inf_height_tree)
        neg_height_tree = ete4.PhyloTree("(A:1,B:1);", parser=1)
        neg_height_tree.children[0].dist = -0.1
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            kfphylo.get_tree_height(neg_height_tree)
        named_tree = kfphylo.fill_internal_node_names("((A:1,B:1):2,C:3);")
        self.assertTrue(all(node.name != "" for node in named_tree.traverse() if (not node.is_leaf)))
        none_named_tree = ete4.PhyloTree("((A:1,B:1):2,C:3);", parser=1)
        for node in none_named_tree.traverse():
            if not node.is_leaf:
                node.name = None
        filled_none_named_tree = kfphylo.fill_internal_node_names(none_named_tree)
        self.assertTrue(all(node.name is not None for node in filled_none_named_tree.traverse() if (not node.is_leaf)))
        self.assertTrue(
            all(str(node.name).strip() != "" for node in filled_none_named_tree.traverse() if (not node.is_leaf))
        )
        with self.assertRaisesRegex(ValueError, "must not be None"):
            kfphylo.add_numerical_node_labels(None)
        with self.assertRaisesRegex(ValueError, "tree must be a Newick string"):
            kfphylo.fill_internal_node_names(0)
        with self.assertRaisesRegex(ValueError, "tree must be a Newick string"):
            kfphylo.check_ultrametric(0)
        with self.assertRaisesRegex(ValueError, "tree must be a Newick string"):
            kfphylo.taxonomic_annotation(0)

    def test_kfphylo_check_ultrametric_zero_length(self):
        tree = ete4.PhyloTree("(A:0,B:0,C:0);", parser=1)
        self.assertTrue(kfphylo.check_ultrametric(tree))
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfphylo.check_ultrametric(tree, tol="bad")
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfphylo.check_ultrametric(tree, tol=np.nan)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            kfphylo.check_ultrametric(tree, tol=-1)
        bad_tree_inf = ete4.PhyloTree("(A:1,B:1);", parser=1)
        bad_tree_inf.children[0].dist = np.inf
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfphylo.check_ultrametric(bad_tree_inf)
        bad_tree_neg = ete4.PhyloTree("(A:1,B:1);", parser=1)
        bad_tree_neg.children[0].dist = -0.1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            kfphylo.check_ultrametric(bad_tree_neg)

    def test_kfphylo_branch_id_is_csubst_compatible(self):
        def _csubst_reference_branch_ids(tree):
            all_leaf_names = sorted(tree.leaf_names())
            leaf_branch_ids = {leaf_name: (1 << i) for i, leaf_name in enumerate(all_leaf_names)}
            nodes = list(tree.traverse())
            clade_signatures = [sum(leaf_branch_ids[leaf_name] for leaf_name in node.leaf_names()) for node in nodes]
            sorted_node_indices = sorted(range(len(nodes)), key=lambda idx: clade_signatures[idx])
            rank_by_node_index = {node_index: rank for rank, node_index in enumerate(sorted_node_indices)}
            return [rank_by_node_index[i] for i in range(len(nodes))]

        tree_small = ete4.PhyloTree("((A:1,B:1):2,C:3);", parser=1)
        expected_small = _csubst_reference_branch_ids(tree_small)
        out_small = kfphylo.add_numerical_node_labels(tree_small)
        actual_small = [node.branch_id for node in out_small.traverse()]
        self.assertEqual(actual_small, expected_small)

        leaf_names = [f"L{i}" for i in range(64)]
        tree_txt = f"{leaf_names[0]}:1"
        for leaf_name in leaf_names[1:]:
            tree_txt = f"({tree_txt},{leaf_name}:1):1"
        tree_large = ete4.PhyloTree(tree_txt + ";", parser=1)
        expected_large = _csubst_reference_branch_ids(tree_large)
        out_large = kfphylo.add_numerical_node_labels(tree_large)
        actual_large = [node.branch_id for node in out_large.traverse()]
        self.assertEqual(actual_large, expected_large)
        dup_leaf_tree = ete4.PhyloTree("((A:1,A:1):1,B:1);", parser=1)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            kfphylo.add_numerical_node_labels(dup_leaf_tree)
        unnamed_leaf_tree = ete4.PhyloTree("(A:1,:1);", parser=1)
        with self.assertRaisesRegex(ValueError, "must be non-empty strings"):
            kfphylo.add_numerical_node_labels(unnamed_leaf_tree)

    def test_kfphylo_load_phylo_tree(self):
        class BadPath(os.PathLike):
            def __fspath__(self):
                return 1

        newick = "((A:1,B:1):2,C:3);"
        tree_from_newick = kfphylo.load_phylo_tree(newick, parser=1)
        self.assertEqual(set(tree_from_newick.leaf_names()), {"A", "B", "C"})
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tree_path = Path(tmp.name)
            tmp.write(newick)
        try:
            tree_from_path = kfphylo.load_phylo_tree(tree_path, parser=1)
            self.assertEqual(set(tree_from_path.leaf_names()), {"A", "B", "C"})
        finally:
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            empty_tree_path = Path(tmp.name)
        try:
            with self.assertRaisesRegex(ValueError, "empty"):
                kfphylo.load_phylo_tree(empty_tree_path, parser=1)
        finally:
            os.unlink(empty_tree_path)
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            invalid_utf8_tree_path = Path(tmp.name)
            tmp.write(b"\xff\xfe\xfd")
        try:
            with self.assertRaisesRegex(ValueError, "Failed to read tree file"):
                kfphylo.load_phylo_tree(invalid_utf8_tree_path, parser=1)
        finally:
            os.unlink(invalid_utf8_tree_path)
        with self.assertRaises(ValueError):
            kfphylo.load_phylo_tree(None, parser=1)
        with self.assertRaises(ValueError):
            kfphylo.load_phylo_tree("   ", parser=1)
        with self.assertRaisesRegex(ValueError, "neither a readable tree file path nor a valid Newick string"):
            kfphylo.load_phylo_tree("not_newick", parser=1)
        with self.assertRaises(TypeError):
            kfphylo.load_phylo_tree(123, parser=1)
        with self.assertRaisesRegex(TypeError, "Newick string, path"):
            kfphylo.load_phylo_tree(BadPath(), parser=1)
        with self.assertRaisesRegex(ValueError, "not a file"):
            kfphylo.load_phylo_tree(Path("."), parser=1)

    def test_kfphylo_transfer_root(self):
        tree_from = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to = ete4.PhyloTree("(A:1,(B:1,(C:1,D:1):2):2);", parser=1)
        out = kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)
        self.assertEqual(set(out.leaf_names()), set(tree_from.leaf_names()))
        self.assertEqual(len(out.get_children()), 2)
        tree_from_inf = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to_inf = ete4.PhyloTree("(A:1,(B:1,(C:1,D:1):2):2);", parser=1)
        tree_from_inf.children[0].dist = np.inf
        with self.assertRaisesRegex(ValueError, "tree_from root child branch lengths must be finite numeric values"):
            kfphylo.transfer_root(tree_to=tree_to_inf, tree_from=tree_from_inf)
        tree_from_none = ete4.PhyloTree("((A,B),(C,D));", parser=1)
        tree_to_none = ete4.PhyloTree("(A:1,(B:1,(C:1,D:1):2):2);", parser=1)
        with self.assertRaisesRegex(ValueError, "tree_from root child branch lengths must be finite numeric values"):
            kfphylo.transfer_root(tree_to=tree_to_none, tree_from=tree_from_none)
        with self.assertRaisesRegex(ValueError, "tree_from leaf names must be non-empty strings"):
            kfphylo.transfer_root(
                tree_to="(A:1,(B:1,C:2):1);",
                tree_from="((:1,B:1):1,C:2);",
            )
        with self.assertRaisesRegex(ValueError, "tree_from leaf names must be unique"):
            kfphylo.transfer_root(
                tree_to="(A:1,(B:1,C:2):1);",
                tree_from="((A:1,A:1):1,C:2);",
            )
        with self.assertRaisesRegex(ValueError, "tree_to must be a Newick string"):
            kfphylo.transfer_root(tree_to=0, tree_from=tree_from)
        with self.assertRaisesRegex(ValueError, "tree_from must be a Newick string"):
            kfphylo.transfer_root(tree_to=tree_to, tree_from=0)
        with self.assertRaisesRegex(ValueError, "verbose must be a boolean value"):
            kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from, verbose="False")

    def test_kfphylo_transfer_root_handles_tree_to_root_distance(self):
        tree_to = ete4.PhyloTree("(S2:0.1,(S3:1,(S1:2,S0:0.1):1):0.1):2;", parser=1)
        tree_from = ete4.PhyloTree("(S0:2,(S3:1,S2:1,S1:2):2):0.1;", parser=1)
        out = kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)
        self.assertEqual(set(out.leaf_names()), {"S0", "S1", "S2", "S3"})
        self.assertEqual(len(out.get_children()), 2)
        self.assertEqual(float(out.dist), 0.0)

    def test_kfphylo_transfer_root_rejects_non_finite_tree_to_root_distance(self):
        tree_to = ete4.PhyloTree("(A:1,(B:1,(C:1,D:1):2):2);", parser=1)
        tree_from = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to.dist = np.inf
        with self.assertRaisesRegex(ValueError, "tree_to root branch length must be a finite numeric value"):
            kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)

    def test_kfphylo_transfer_root_requires_bifurcating_root(self):
        tree_from = "(A:1,B:1,C:1);"
        tree_to = "((A:1,B:1):1,C:2);"
        with self.assertRaisesRegex(ValueError, "bifurcating"):
            kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)

    def test_kfphylo_transfer_root_accepts_multifurcating_tree_to_root(self):
        tree_from = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to = ete4.PhyloTree("(A:1,B:1,(C:1,D:1):2);", parser=1)
        out = kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)
        self.assertEqual(set(out.leaf_names()), set(tree_from.leaf_names()))
        self.assertEqual(len(out.get_children()), 2)

    def test_kfphylo_transfer_root_raises_on_incompatible_split(self):
        tree_from = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to = ete4.PhyloTree("(A:1,C:1,(B:1,D:1):2);", parser=1)
        tree_to.dist = 5.0
        with self.assertRaisesRegex(ValueError, "root split"):
            kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)
        self.assertEqual(float(tree_to.dist), 5.0)

    def test_kfphylo_transfer_root_raises_on_tip_mismatch(self):
        tree_from = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to = ete4.PhyloTree("((A:1,B:1):2,(C:1,E:1):2);", parser=1)
        with self.assertRaisesRegex(ValueError, "identical tips"):
            kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)

    def test_kfphylo_transfer_internal_node_names_requires_same_topology(self):
        tree_from = "((A:1,B:1):2,(C:1,D:1):2);"
        tree_to = "((A:1,C:1):2,(B:1,D:1):2);"
        with self.assertRaisesRegex(ValueError, "RF distance"):
            kfphylo.transfer_internal_node_names(tree_to=tree_to, tree_from=tree_from)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            kfphylo.transfer_internal_node_names(
                tree_to="((A:1,B:1):2,(C:1,D:1):2);",
                tree_from="((A:1,A:1):2,(C:1,D:1):2);",
            )
        with self.assertRaisesRegex(ValueError, "tree_to must be a Newick string"):
            kfphylo.transfer_internal_node_names(tree_to=0, tree_from=tree_from)
        with self.assertRaisesRegex(ValueError, "tree_from must be a Newick string"):
            kfphylo.transfer_internal_node_names(tree_to=tree_to, tree_from=0)

    def test_kfphylo_taxonomic_annotation_validates_leaf_names(self):
        tree = ete4.PhyloTree("(A:1,B_c:1);", parser=1)
        with self.assertRaisesRegex(ValueError, "genus and species"):
            kfphylo.taxonomic_annotation(tree)
        tree_none_leaf = ete4.PhyloTree("(A_b:1,C_d:1);", parser=1)
        tree_none_leaf.children[0].name = None
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            kfphylo.taxonomic_annotation(tree_none_leaf)

    def test_kfphylo_taxonomic_annotation_handles_ncbi_failures(self):
        class DummyNcbiGetTranslatorFailure:
            def get_name_translator(self, names):
                raise RuntimeError("boom")

        class DummyNcbiAnnotateFailure:
            def get_name_translator(self, names):
                return {"Homo sapiens": [9606], "Mus musculus": [10090]}

            def annotate_tree(self, tree, taxid_attr="taxid"):
                raise RuntimeError("boom")

        tree = ete4.PhyloTree("(Homo_sapiens:1,Mus_musculus:1);", parser=1)
        with (
            mock.patch("kftools.kfphylo.ete4.NCBITaxa", side_effect=RuntimeError("boom")),
            self.assertRaisesRegex(ValueError, "Failed to initialize NCBITaxa database"),
        ):
            kfphylo.taxonomic_annotation(tree)
        with (
            mock.patch("kftools.kfphylo.ete4.NCBITaxa", return_value=DummyNcbiGetTranslatorFailure()),
            self.assertRaisesRegex(ValueError, "Failed to query scientific names in NCBITaxa"),
        ):
            kfphylo.taxonomic_annotation(tree)
        with (
            mock.patch("kftools.kfphylo.ete4.NCBITaxa", return_value=DummyNcbiAnnotateFailure()),
            self.assertRaisesRegex(ValueError, "Failed to annotate tree with NCBI taxonomy"),
        ):
            kfphylo.taxonomic_annotation(tree)

    def test_kfphylo_taxonomic_annotation_uses_species_parser_without_rewriting_labels(self):
        class DummyNcbi:
            def __init__(self):
                self.names = None
                self.annotated = False

            def get_name_translator(self, names):
                self.names = sorted(names)
                return {"Amoeba": [2812], "Dictyostelium discoideum": [5786]}

            def annotate_tree(self, tree, taxid_attr="taxid"):
                self.annotated = True

        tree = ete4.PhyloTree(
            "(Dictyostelium_cf_discoideum|gene1:1,Amoeba_sp_JDSRuffled|gene2:1);",
            parser=1,
        )
        dummy_ncbi = DummyNcbi()
        with mock.patch("kftools.kfphylo.ete4.NCBITaxa", return_value=dummy_ncbi):
            out = kfphylo.taxonomic_annotation(tree, species_parser="taxonomic")
        self.assertEqual(dummy_ncbi.names, ["Amoeba", "Dictyostelium discoideum"])
        self.assertTrue(dummy_ncbi.annotated)
        self.assertEqual(
            [leaf.name for leaf in out.leaves()],
            ["Dictyostelium_cf_discoideum|gene1", "Amoeba_sp_JDSRuffled|gene2"],
        )
        self.assertEqual(
            [leaf.sci_name for leaf in out.leaves()],
            ["Dictyostelium cf. discoideum", "Amoeba sp. JDSRuffled"],
        )
        self.assertEqual(
            [leaf.taxonomy_query for leaf in out.leaves()],
            ["Dictyostelium discoideum", "Amoeba"],
        )
