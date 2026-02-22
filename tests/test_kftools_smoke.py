import gzip
import io
import os
import tempfile
import unittest
from unittest import mock
import contextlib
import warnings
from pathlib import Path

import ete4
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as stats
matplotlib.use("Agg")

from kftools import kfexpression
from kftools import kfog
from kftools import kfphylo
from kftools import kfplot
from kftools import kfseq
from kftools import kfstat
from kftools import kfutil


class TestKFToolsSmoke(unittest.TestCase):
    def test_kfutil(self):
        self.assertEqual(kfutil.add_dict_key_prefix({"a": 1}, "p"), {"p_a": 1})
        self.assertEqual(kfutil.add_dict_key_prefix({1: 2}, "p"), {"p_1": 2})
        with self.assertRaisesRegex(ValueError, "mapping type"):
            kfutil.add_dict_key_prefix([("a", 1)], "p")
        with self.assertRaisesRegex(ValueError, "prefix must be a string"):
            kfutil.add_dict_key_prefix({"a": 1}, 1)
        self.assertEqual(kfutil.rgb_to_hex(1, 0.5, 0), "#FF8000")
        self.assertEqual(len(kfutil.get_rgb_gradient(5, [1, 0, 0], [0, 0, 1])), 5)
        grad3 = kfutil.get_rgb_gradient(3, [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5])
        self.assertEqual(grad3[0], [1.0, 0.0, 0.0])
        self.assertEqual(grad3[1], [0.5, 0.5, 0.5])
        self.assertEqual(grad3[2], [0.0, 0.0, 1.0])
        grad2 = kfutil.get_rgb_gradient(2, [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5])
        self.assertEqual(grad2[0], [1.0, 0.0, 0.0])
        self.assertEqual(grad2[1], [0.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "ncol must be an integer"):
            kfutil.get_rgb_gradient(2.5, [1, 0, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "exactly 3 channel values"):
            kfutil.get_rgb_gradient(3, [1, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "exactly 3 channel values"):
            kfutil.get_rgb_gradient(3, object(), [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfutil.get_rgb_gradient(3, [2, 0, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "bool is not allowed"):
            kfutil.get_rgb_gradient(3, [True, 0, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "bool is not allowed"):
            kfutil.get_rgb_gradient(3, [1, 0, 0], [0, False, 1])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfutil.rgb_to_hex(-0.1, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfutil.rgb_to_hex(1.1, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            kfutil.rgb_to_hex("a", 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            kfutil.rgb_to_hex(np.nan, 0.0, 0.0)

    def test_kfexpression(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
        tau = kfexpression.calc_tau(df, ["a", "b"], unlog2=False, unPlus1=False)
        self.assertEqual(len(tau), 2)
        self.assertTrue(np.isfinite(tau).all())
        tau_single_col = kfexpression.calc_tau(df, "a", unlog2=False, unPlus1=False)
        self.assertEqual(len(tau_single_col), 2)
        self.assertAlmostEqual(kfexpression.calc_complementarity([1, 2], [1, 1]), 0.25)
        self.assertAlmostEqual(kfexpression.calc_complementarity([1, 2, 3], [1]), 0.0)
        with self.assertRaisesRegex(ValueError, "at least one value"):
            kfexpression.calc_complementarity([], [1])
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfexpression.calc_complementarity([1, {}], [1, 2])
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfexpression.calc_complementarity([1, np.inf], [1, 2])
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfexpression.calc_complementarity([1, 2], [1, np.nan])
        with self.assertRaisesRegex(ValueError, "at least one"):
            kfexpression.calc_tau(df, [], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "not found"):
            kfexpression.calc_tau(df, ["z"], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfexpression.calc_tau(pd.DataFrame({"a": ["x"]}), ["a"])
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            kfexpression.calc_tau(df, ["a", "a"], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "non-empty string column names"):
            kfexpression.calc_tau(df, [{}], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "non-empty string column names"):
            kfexpression.calc_tau(df, [""], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfexpression.calc_tau(
                pd.DataFrame({"a": [1.0, np.inf], "b": [2.0, 3.0]}),
                ["a", "b"],
                unlog2=False,
                unPlus1=False,
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tau_zero = kfexpression.calc_tau(
                pd.DataFrame({"a": [0.0, 0.0]}),
                ["a"],
                unlog2=False,
                unPlus1=False,
            )
            self.assertEqual(tau_zero.tolist(), [0.0, 0.0])
            self.assertFalse(
                any("invalid value encountered in divide" in str(wi.message) for wi in w),
                "calc_tau should avoid runtime warnings on zero-max rows",
            )
        with self.assertRaisesRegex(ValueError, "non-empty sequence"):
            kfexpression.calc_tau(df, 0)
        with self.assertRaisesRegex(ValueError, "DataFrame-like"):
            kfexpression.calc_tau(None, ["a"])
        with self.assertRaisesRegex(ValueError, "unlog2 must be a boolean value"):
            kfexpression.calc_tau(df, ["a", "b"], unlog2="False", unPlus1=False)
        with self.assertRaisesRegex(ValueError, "unPlus1 must be a boolean value"):
            kfexpression.calc_tau(df, ["a", "b"], unlog2=True, unPlus1="False")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(ValueError, "out of range"):
                kfexpression.calc_tau(
                    pd.DataFrame({"a": [2000.0], "b": [2000.0]}),
                    ["a", "b"],
                    unlog2=True,
                    unPlus1=True,
                )
            self.assertFalse(
                any("overflow encountered in exp2" in str(wi.message) for wi in w),
                "calc_tau should not leak exp2 overflow warnings",
            )

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
        self.assertTrue(all(str(node.name).strip() != "" for node in filled_none_named_tree.traverse() if (not node.is_leaf)))
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
            clade_signatures = [
                sum(leaf_branch_ids[leaf_name] for leaf_name in node.leaf_names())
                for node in nodes
            ]
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
        with mock.patch("kftools.kfphylo.ete4.NCBITaxa", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(ValueError, "Failed to initialize NCBITaxa database"):
                kfphylo.taxonomic_annotation(tree)
        with mock.patch("kftools.kfphylo.ete4.NCBITaxa", return_value=DummyNcbiGetTranslatorFailure()):
            with self.assertRaisesRegex(ValueError, "Failed to query scientific names in NCBITaxa"):
                kfphylo.taxonomic_annotation(tree)
        with mock.patch("kftools.kfphylo.ete4.NCBITaxa", return_value=DummyNcbiAnnotateFailure()):
            with self.assertRaisesRegex(ValueError, "Failed to annotate tree with NCBI taxonomy"):
                kfphylo.taxonomic_annotation(tree)

    def test_kfseq(self):
        codon_freqs = {"AAA": 0.5, "TTT": 0.5}
        out = kfseq.codon2nuc_freqs(codon_freqs=codon_freqs, model="F3X4")
        self.assertEqual(len(out), 3)
        with self.assertRaises(ValueError):
            kfseq.codon2nuc_freqs(codon_freqs=codon_freqs, model="HKY")
        tree = ete4.PhyloTree("(A:2,B:1);", parser=1)
        subroot_thetas = {
            "A": [{"theta": 0.1, "theta1": 0.3, "theta2": 0.7}] * 3,
            "B": [{"theta": 0.9, "theta1": 0.7, "theta2": 0.3}] * 3,
        }
        root_thetas = kfseq.weighted_mean_root_thetas(subroot_thetas, tree, model="F3X4")
        self.assertEqual(len(root_thetas), 3)
        self.assertAlmostEqual(root_thetas[0]["theta"], 0.1 + (0.9 - 0.1) * (2.0 / 3.0))

        tree3 = ete4.PhyloTree("(A:1,B:2,C:3);", parser=1)
        subroot_thetas3 = {
            "A": [{"theta": 0.1}] * 3,
            "B": [{"theta": 0.5}] * 3,
            "C": [{"theta": 0.9}] * 3,
        }
        root_thetas3 = kfseq.weighted_mean_root_thetas(subroot_thetas3, tree3, model="F3X4")
        self.assertEqual(len(root_thetas3), 3)
        self.assertAlmostEqual(root_thetas3[0]["theta"], 0.1 + (0.9 - 0.1) * (1.0 / 4.0))
        self.assertEqual(kfseq.get_mapnh_thetas("F3X4", []), "F3X4()")

    def test_kfseq_robustness_guards(self):
        class BadPath(os.PathLike):
            def __fspath__(self):
                return 1

        with self.assertRaisesRegex(ValueError, "positive total"):
            kfseq.codon2nuc_freqs(codon_freqs={}, model="F3X4")
        with self.assertRaisesRegex(ValueError, "model must be a string"):
            kfseq.codon2nuc_freqs(codon_freqs={"AAA": 1.0}, model=None)
        with self.assertRaisesRegex(ValueError, "must be a mapping"):
            kfseq.codon2nuc_freqs(codon_freqs=["AAA", 1.0], model="F3X4")
        with self.assertRaisesRegex(ValueError, "model must be a string"):
            kfseq.get_mapnh_thetas(None, [])
        with self.assertRaisesRegex(ValueError, "model must be a string"):
            kfseq.alignment2nuc_freqs("A", __file__, None)
        with self.assertRaisesRegex(ValueError, "path-like"):
            kfseq.alignment2nuc_freqs("A", 1.2, "F3X4")
        with self.assertRaisesRegex(ValueError, "path-like"):
            kfseq.alignment2nuc_freqs("A", BadPath(), "F3X4")
        with self.assertRaisesRegex(ValueError, "bytes are not supported"):
            kfseq.alignment2nuc_freqs("A", b"/tmp/definitely_missing_kftools_alignment_123456.fa", "F3X4")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            kfseq.alignment2nuc_freqs(None, __file__, "F3X4")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            kfseq.alignment2nuc_freqs("", __file__, "F3X4")
        with self.assertRaisesRegex(ValueError, "Failed to read alignment_file"):
            kfseq.alignment2nuc_freqs("A", "/tmp/definitely_missing_kftools_alignment_123456.fa", "F3X4")

        lower_out = kfseq.codon2nuc_freqs(codon_freqs={"aaa": 1.0}, model="F3X4")
        self.assertEqual(lower_out[0]["A"], 1.0)
        self.assertEqual(lower_out[1]["A"], 1.0)
        self.assertEqual(lower_out[2]["A"], 1.0)
        with self.assertRaisesRegex(ValueError, "invalid nucleotides"):
            kfseq.codon2nuc_freqs(codon_freqs={"AAN": 1.0}, model="F3X4")
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfseq.codon2nuc_freqs(codon_freqs={"AAA": "1.0"}, model="F3X4")
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfseq.codon2nuc_freqs(codon_freqs={"AAA": True}, model="F3X4")
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfseq.codon2nuc_freqs(codon_freqs={"AAA": np.nan}, model="F3X4")
        with self.assertRaisesRegex(ValueError, "missing keys"):
            kfseq.get_mapnh_thetas("F3X4", [{"theta": 0.1}])
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfseq.get_mapnh_thetas("F3X4", [{"theta": "x", "theta1": 0.5, "theta2": 0.5}])
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfseq.get_mapnh_thetas("F3X4", [{"theta": True, "theta1": 0.5, "theta2": 0.5}])
        with self.assertRaisesRegex(ValueError, "missing keys"):
            kfseq.nuc_freq2theta([{"A": 1.0, "T": 1.0}])
        with self.assertRaisesRegex(ValueError, "list or tuple"):
            kfseq.nuc_freq2theta("ATCG")
        with self.assertRaisesRegex(ValueError, "must be a dictionary"):
            kfseq.nuc_freq2theta(["ATCG"])
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfseq.nuc_freq2theta([{"A": "x", "T": 1.0, "C": 0.0, "G": 0.0}])
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfseq.nuc_freq2theta([{"A": True, "T": 1.0, "C": 0.0, "G": 0.0}])

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            aln_path = tmp.name
            tmp.write(">leafA\nA\n")
        try:
            with self.assertRaisesRegex(ValueError, "at least three nucleotides"):
                kfseq.alignment2nuc_freqs("leafA", aln_path, "F3X4")
        finally:
            os.unlink(aln_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            aln_empty_target = tmp.name
            tmp.write(">leafA\n")
            tmp.write(">leafB\nATGATG\n")
        try:
            with self.assertRaisesRegex(ValueError, "is empty in alignment_file"):
                kfseq.alignment2nuc_freqs("leafA", aln_empty_target, "F3X4")
        finally:
            os.unlink(aln_empty_target)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            aln_ambiguous = tmp.name
            tmp.write(">AA\nATGATG\n")
            tmp.write(">A\nTTTTTT\n")
        try:
            out = kfseq.alignment2nuc_freqs("A", aln_ambiguous, "F3X4")
            self.assertEqual(out[0]["T"], 1.0)
            self.assertEqual(out[1]["T"], 1.0)
            self.assertEqual(out[2]["T"], 1.0)
        finally:
            os.unlink(aln_ambiguous)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            aln_duplicate_target = tmp.name
            tmp.write(">A\nATGATG\n")
            tmp.write(">A\nTTTTTT\n")
        try:
            with self.assertRaisesRegex(ValueError, "appears multiple times"):
                kfseq.alignment2nuc_freqs("A", aln_duplicate_target, "F3X4")
        finally:
            os.unlink(aln_duplicate_target)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            aln_invalid_nuc = tmp.name
            tmp.write(">A\nANNANN\n")
        try:
            with self.assertRaisesRegex(ValueError, "invalid nucleotides"):
                kfseq.alignment2nuc_freqs("A", aln_invalid_nuc, "F3X4")
        finally:
            os.unlink(aln_invalid_nuc)

        tree_missing = ete4.PhyloTree("(A:1,B:1);", parser=1)
        subroot_thetas_missing = {"A": [{"theta": 0.2}] * 3}
        with self.assertRaisesRegex(ValueError, "missing node"):
            kfseq.weighted_mean_root_thetas(subroot_thetas_missing, tree_missing, model="F3X4")
        theta_template_a = [{"theta": 0.2, "theta1": 0.5, "theta2": 0.5} for _ in range(3)]
        theta_template_b = [{"theta": 0.8, "theta1": 0.5, "theta2": 0.5} for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "unknown node names"):
            kfseq.weighted_mean_root_thetas(
                {"A": theta_template_a, "B": theta_template_b, "C": theta_template_a},
                tree_missing,
                model="F3X4",
            )
        with self.assertRaisesRegex(ValueError, "must contain 3"):
            kfseq.weighted_mean_root_thetas(
                {"A": [{"theta": 0.2, "theta1": 0.5, "theta2": 0.5}], "B": theta_template_b},
                tree_missing,
                model="F3X4",
            )
        with self.assertRaisesRegex(ValueError, "identical parameter keys"):
            kfseq.weighted_mean_root_thetas(
                {
                    "A": theta_template_a,
                    "B": [{"theta": 0.8, "theta1": 0.5} for _ in range(3)],
                },
                tree_missing,
                model="F3X4",
            )
        unnamed_tree = ete4.PhyloTree("((A:1,B:1):1,(C:1,D:1):1);", parser=1)
        with self.assertRaisesRegex(ValueError, "non-empty names"):
            kfseq.weighted_mean_root_thetas({}, unnamed_tree, model="F3X4")
        duplicate_name_tree = ete4.PhyloTree("((A:1,B:1)X:1,(C:1,D:1)X:1);", parser=1)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            kfseq.weighted_mean_root_thetas(
                {"X": [{"theta": 0.1, "theta1": 0.5, "theta2": 0.5}] * 3},
                duplicate_name_tree,
                model="F3X4",
            )
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfseq.weighted_mean_root_thetas(
                {
                    "A": [{"theta": "x", "theta1": 0.5, "theta2": 0.5}] * 3,
                    "B": theta_template_b,
                },
                tree_missing,
                model="F3X4",
            )
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfseq.weighted_mean_root_thetas(
                {
                    "A": [{"theta": True, "theta1": 0.5, "theta2": 0.5}] * 3,
                    "B": theta_template_b,
                },
                tree_missing,
                model="F3X4",
            )
        with self.assertRaisesRegex(ValueError, "must not be None"):
            kfseq.weighted_mean_root_thetas({}, None, model="F3X4")
        with self.assertRaisesRegex(ValueError, "dictionary keyed by subroot"):
            kfseq.weighted_mean_root_thetas([], tree_missing, model="F3X4")
        tree_negative_bl = ete4.PhyloTree("(A:1,B:1);", parser=1)
        tree_negative_bl.children[0].dist = -1.0
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            kfseq.weighted_mean_root_thetas(
                {"A": theta_template_a, "B": theta_template_b},
                tree_negative_bl,
                model="F3X4",
            )
        tree_nonfinite_bl = ete4.PhyloTree("(A:1,B:1);", parser=1)
        tree_nonfinite_bl.children[0].dist = np.inf
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfseq.weighted_mean_root_thetas(
                {"A": theta_template_a, "B": theta_template_b},
                tree_nonfinite_bl,
                model="F3X4",
            )

    def test_kfseq_weighted_mean_root_thetas_zero_branch_lengths(self):
        tree2 = ete4.PhyloTree("(A:0,B:0);", parser=1)
        subroot_thetas2 = {
            "A": [{"theta": 0.2}] * 3,
            "B": [{"theta": 0.8}] * 3,
        }
        root_thetas2 = kfseq.weighted_mean_root_thetas(subroot_thetas2, tree2, model="F3X4")
        self.assertAlmostEqual(root_thetas2[0]["theta"], 0.5)

        tree3 = ete4.PhyloTree("(A:0,B:0,C:0);", parser=1)
        subroot_thetas3 = {
            "A": [{"theta": 0.1}] * 3,
            "B": [{"theta": 0.3}] * 3,
            "C": [{"theta": 0.9}] * 3,
        }
        root_thetas3 = kfseq.weighted_mean_root_thetas(subroot_thetas3, tree3, model="F3X4")
        self.assertAlmostEqual(root_thetas3[0]["theta"], 0.5)

    def test_kfseq_weighted_mean_root_thetas_is_independent_of_dict_order(self):
        tree3 = ete4.PhyloTree("(A:1,B:2,C:3);", parser=1)
        a_theta = [{"theta": 0.1}] * 3
        b_theta = [{"theta": 0.1}] * 3
        c_theta = [{"theta": 0.9}] * 3
        subroot_thetas_abc = {"A": a_theta, "B": b_theta, "C": c_theta}
        subroot_thetas_bac = {"B": b_theta, "A": a_theta, "C": c_theta}
        root_abc = kfseq.weighted_mean_root_thetas(subroot_thetas_abc, tree3, model="F3X4")
        root_bac = kfseq.weighted_mean_root_thetas(subroot_thetas_bac, tree3, model="F3X4")
        self.assertAlmostEqual(root_abc[0]["theta"], root_bac[0]["theta"])

    def test_kfseq_weighted_mean_root_thetas_is_independent_of_tree_child_order_with_tied_extrema(self):
        tree_abc = ete4.PhyloTree("(A:1,B:3,C:4);", parser=1)
        tree_bac = ete4.PhyloTree("(B:3,A:1,C:4);", parser=1)
        subroot_thetas = {
            "A": [{"theta": 0.1}] * 3,
            "B": [{"theta": 0.1}] * 3,
            "C": [{"theta": 0.9}] * 3,
        }
        root_abc = kfseq.weighted_mean_root_thetas(subroot_thetas, tree_abc, model="F3X4")
        root_bac = kfseq.weighted_mean_root_thetas(subroot_thetas, tree_bac, model="F3X4")
        self.assertAlmostEqual(root_abc[0]["theta"], root_bac[0]["theta"])

    def test_kfog(self):
        newick = "((A_a:1,B_b:1):1,C_c:2);"
        df = kfog.nwk2table(newick, attr="dist", age=True)
        self.assertGreater(len(df), 0)
        df_name = kfog.nwk2table(
            "((Alpha_one:1,Beta_two:1)NodeX:1,Gamma_three:2)RootName;",
            attr="name",
            age=False,
        )
        self.assertEqual(
            set(df_name["name"].tolist()),
            {"Alpha_one", "Beta_two", "Gamma_three", "NodeX", "RootName"},
        )
        df2 = kfog.nwk2table(newick, attr="dist", age=False, parent=True, sister=True)
        self.assertIn("parent", df2.columns)
        self.assertIn("sister", df2.columns)
        self.assertEqual(df2["branch_id"].tolist(), sorted(df2["branch_id"].tolist()))
        mixed_attr_tree = ete4.PhyloTree("((A_a:1,B_b:1):1,C_c:2);", parser=1)
        for node in mixed_attr_tree.traverse():
            node.custom_attr = 1
        for leaf in mixed_attr_tree.leaves():
            if leaf.name == "C_c":
                leaf.custom_attr = None
        mixed_attr_table = kfog.nwk2table(mixed_attr_tree, attr="custom_attr")
        self.assertEqual(
            mixed_attr_table.shape[0],
            len(list(mixed_attr_tree.traverse())),
        )
        self.assertIn(None, mixed_attr_table["custom_attr"].tolist())
        with self.assertRaisesRegex(ValueError, "attr must be a string"):
            kfog.nwk2table(newick, attr=None)
        with self.assertRaisesRegex(ValueError, "tree must be a Newick string"):
            kfog.nwk2table(0, attr="dist")
        with self.assertRaisesRegex(ValueError, "valid Newick string"):
            kfog.nwk2table("not_newick", attr="dist")

    def test_kfog_nwk2table_age_requires_ultrametric(self):
        non_ultrametric = "((A_a:1,B_b:2):1,C_c:2);"
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "ultrametric"):
                kfog.nwk2table(non_ultrametric, attr="dist", age=True)
        with self.assertRaisesRegex(ValueError, "only when attr='dist'"):
            kfog.nwk2table("((A_a:1,B_b:1):1,C_c:2);", attr="support", age=True)
        with self.assertRaisesRegex(ValueError, "age must be a boolean value"):
            kfog.nwk2table("((A_a:1,B_b:1):1,C_c:2);", attr="dist", age="False")
        with self.assertRaisesRegex(ValueError, "parent must be a boolean value"):
            kfog.nwk2table("((A_a:1,B_b:1):1,C_c:2);", attr="dist", parent="False")
        with self.assertRaisesRegex(ValueError, "sister must be a boolean value"):
            kfog.nwk2table("((A_a:1,B_b:1):1,C_c:2);", attr="dist", sister="False")

    def test_kfog_nwk2table_pathlike_input(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tree_path = Path(tmp.name)
            tmp.write("((A_a:1,B_b:1):1,C_c:2);")
        try:
            df = kfog.nwk2table(tree_path, attr="dist")
            self.assertGreater(len(df), 0)
        finally:
            os.unlink(tree_path)

    def test_kfog_misc_node_statistics(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write("((S1_a_1:1,S1_a_2:1):1,S2_b_1:2);")
        try:
            out = kfog.get_misc_node_statistics(path, tax_annot=False)
            self.assertIn("num_sp", out.columns)
            self.assertIn("dup_conf_score", out.columns)
            root = out.loc[out["parent"] == -999, :].iloc[0]
            self.assertEqual(int(root["num_sp"]), 2)
            self.assertTrue((out["so_event"] == "D").any())
        finally:
            os.unlink(path)
        with self.assertRaisesRegex(ValueError, "tree_file must be a Newick string"):
            kfog.get_misc_node_statistics(0, tax_annot=False)
        with self.assertRaisesRegex(ValueError, "tax_annot must be a boolean value"):
            kfog.get_misc_node_statistics("((S1_a_1:1,S1_a_2:1):1,S2_b_1:2);", tax_annot="False")
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            poly_path = tmp.name
            tmp.write("(S1_a_1:1,S1_a_2:1,S1_a_3:1,S2_b_1:1);")
        try:
            out_poly = kfog.get_misc_node_statistics(poly_path, tax_annot=False)
            poly_root = out_poly.loc[out_poly["parent"] == -999, :].iloc[0]
            self.assertAlmostEqual(float(poly_root["dup_conf_score"]), 0.5)
            leaf_rows = out_poly.loc[out_poly["num_leaf"] == 1, :]
            self.assertTrue((leaf_rows["sister"] != -999).all())
        finally:
            os.unlink(poly_path)

    def test_kfog_iqtree_stats(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            path = tmp.name
        try:
            with gzip.open(path, "wb") as f:
                f.write(b"best_model_AIC: M1\n")
            out = kfog.get_iqtree_model_stats(path)
            self.assertEqual(out["iqtree_best_AIC"], "M1")
        finally:
            os.unlink(path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            non_gzip_path = tmp.name
            tmp.write("best_model_AIC: M1\n")
        try:
            with self.assertRaisesRegex(ValueError, "gzip"):
                kfog.get_iqtree_model_stats(non_gzip_path)
        finally:
            os.unlink(non_gzip_path)
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            invalid_utf8_path = tmp.name
        try:
            with gzip.open(invalid_utf8_path, "wb") as f:
                f.write(b"\xff\xfe\xfd")
            with self.assertRaisesRegex(ValueError, "UTF-8"):
                kfog.get_iqtree_model_stats(invalid_utf8_path)
        finally:
            os.unlink(invalid_utf8_path)
        with self.assertRaisesRegex(ValueError, "path-like"):
            kfog.get_iqtree_model_stats(ete4.PhyloTree("(A:1,B:1);", parser=1))

    def test_kfog_file_argument_validation(self):
        class BadPath(os.PathLike):
            def __fspath__(self):
                return 1

        bad_file = ete4.PhyloTree("(A:1,B:1);", parser=1)
        file_funcs = [
            kfog.get_notung_root_stats,
            kfog.get_notung_reconcil_stats,
            kfog.get_root_stats,
            kfog.get_aln_stats,
            kfog.get_dating_method,
            kfog.regime2tree,
        ]
        for fn in file_funcs:
            with self.assertRaisesRegex(ValueError, "path-like"):
                fn(bad_file)
        for fn in file_funcs:
            with self.assertRaisesRegex(ValueError, "path-like"):
                fn(BadPath())
        for fn in file_funcs:
            with self.assertRaisesRegex(ValueError, "bytes are not supported"):
                fn(b"/tmp/definitely_missing_kftools_file_123456789.txt")
        for fn in file_funcs:
            with self.assertRaisesRegex(ValueError, "Failed to read file"):
                fn("/tmp/definitely_missing_kftools_file_123456789.txt")
        with self.assertRaisesRegex(ValueError, "regime_file must be a path-like"):
            kfog.ou2table(bad_file, "x.tsv", "x.nwk")
        with self.assertRaisesRegex(ValueError, "regime_file must be a path-like"):
            kfog.ou2table(BadPath(), "x.tsv", "x.nwk")
        with self.assertRaisesRegex(ValueError, "bytes are not supported"):
            kfog.ou2table(
                b"/tmp/definitely_missing_kftools_regime_123456.tsv",
                "x.tsv",
                "x.nwk",
            )
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("(A_x:1,B_x:1)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "Failed to read regime_file"):
                kfog.ou2table("/tmp/definitely_missing_kftools_regime_123456.tsv", "x.tsv", tree_path)
        finally:
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("wb", delete=False) as regime_tmp:
            regime_bad_utf8_path = regime_tmp.name
            regime_tmp.write(b"\xff\xfe\xfd")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("(A_x:1,B_x:1)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "UTF-8 tab-separated text"):
                kfog.ou2table(regime_bad_utf8_path, leaf_path, tree_path)
        finally:
            os.unlink(regime_bad_utf8_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)

    def test_kfog_node_gene2species_ultrametric(self):
        species_tree = ete4.PhyloTree("((A_x:1,B_x:1):1,(C_x:1,D_x:1):1);", parser=1)
        gene_tree = ete4.PhyloTree("((A_x_g1:1,B_x_g2:1):1,(C_x_g3:1,D_x_g4:1):1);", parser=1)
        out = kfog.node_gene2species(gene_tree, species_tree, is_ultrametric=True)
        self.assertIn("spnode_coverage", out.columns)
        self.assertIn("spnode_age", out.columns)
        self.assertEqual(len(out), len(list(gene_tree.traverse())))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "species_tree must be ultrametric when is_ultrametric=True"):
                kfog.node_gene2species(
                    gene_tree,
                    "((A_x:1,B_x:2):1,(C_x:1,D_x:1):1);",
                    is_ultrametric=True,
                )
        species_tree_bad_dist = ete4.PhyloTree("((A_x:1,B_x:1):1,(C_x:1,D_x:1):1);", parser=1)
        species_tree_bad_dist.children[0].dist = None
        with self.assertRaisesRegex(ValueError, "finite non-negative branch lengths"):
            kfog.node_gene2species(gene_tree, species_tree_bad_dist, is_ultrametric=True)

    def test_kfog_node_gene2species_validates_gene_leaf_name(self):
        species_tree = "((A_x:1,B_x:1):1,C_x:2);"
        gene_tree = "((A_x_g1:1,Bx:1):1,C_x_g3:2);"
        with self.assertRaisesRegex(ValueError, "Gene leaf name"):
            kfog.node_gene2species(gene_tree, species_tree, is_ultrametric=False)
        with self.assertRaisesRegex(ValueError, "species_tree leaf names must be non-empty strings"):
            kfog.node_gene2species(
                "((A_x_g1:1,B_x_g2:1):1,C_x_g3:2);",
                "((:1,B_x:1):1,C_x:2);",
                is_ultrametric=False,
            )
        species_tree_none_name = ete4.PhyloTree("((A_x:1,B_x:1):1,C_x:2);", parser=1)
        species_tree_none_name.children[0].children[0].name = None
        with self.assertRaisesRegex(ValueError, "species_tree leaf names must be non-empty strings"):
            kfog.node_gene2species(
                "((A_x_g1:1,B_x_g2:1):1,C_x_g3:2);",
                species_tree_none_name,
                is_ultrametric=False,
            )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            kfog.node_gene2species(
                "((A_x_g1:1,A_x_g2:1):1,C_x_g3:2);",
                "((A_x:1,A_x:1):1,C_x:2);",
                is_ultrametric=False,
            )
        with self.assertRaisesRegex(ValueError, "gene_tree must be a Newick string"):
            kfog.node_gene2species(0, species_tree, is_ultrametric=False)
        with self.assertRaisesRegex(ValueError, "species_tree must be a Newick string"):
            kfog.node_gene2species(gene_tree, 0, is_ultrametric=False)
        with self.assertRaisesRegex(ValueError, "is_ultrametric must be a boolean value"):
            kfog.node_gene2species(gene_tree, species_tree, is_ultrametric="False")

    def test_kfog_ou2table(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\tt2\n")
            leaf_tmp.write("x\tmu\t0\t1.0\t2.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\t3.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            out = kfog.ou2table(regime_path, leaf_path, tree_path)
            self.assertIn("branch_id", out.columns)
            self.assertIn("tau", out.columns)
            self.assertIn("delta_tau", out.columns)
            self.assertIn("mu_t1", out.columns)
            self.assertEqual(out.shape[0], len(list(kfphylo.load_phylo_tree(tree_path, parser=1).traverse())))
        finally:
            os.unlink(regime_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)

    def test_kfog_ou2table_accepts_shuffled_leaf_trait_columns(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("t1\tnode_name\tregime\tparam\tt2\n")
            leaf_tmp.write("1.0\tx\t0\tmu\t2.0\n")
            leaf_tmp.write("2.0\tx\t1\tmu\t3.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            out = kfog.ou2table(regime_path, leaf_path, tree_path)
            self.assertIn("mu_t1", out.columns)
            self.assertIn("mu_t2", out.columns)
            self.assertNotIn("mu_param", out.columns)
            self.assertEqual(out.shape[0], len(list(kfphylo.load_phylo_tree(tree_path, parser=1).traverse())))
        finally:
            os.unlink(regime_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)

    def test_kfog_ou2table_requires_mu_for_all_regimes(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\tt2\n")
            leaf_tmp.write("x\tmu\t0\t1.0\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "Missing mu values"):
                kfog.ou2table(regime_path, leaf_path, tree_path)
        finally:
            os.unlink(regime_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)

    def test_kfog_get_aln_stats(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write(">s1\nA-CG\n")
            tmp.write(">s2\nAT-G\n")
        try:
            out = kfog.get_aln_stats(path)
            self.assertEqual(out["num_site"], 4)
            self.assertEqual(out["num_seq"], 2)
            self.assertEqual(out["len_max"], 3)
            self.assertEqual(out["len_min"], 3)
        finally:
            os.unlink(path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            invalid_path = tmp.name
            tmp.write("A-CG\n")
        try:
            with self.assertRaisesRegex(ValueError, "FASTA-formatted"):
                kfog.get_aln_stats(invalid_path)
        finally:
            os.unlink(invalid_path)

    def test_kfog_parsers_and_lookup(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write("Number of optimal roots: 2 out of 10\n")
            tmp.write("Best rooting score: 1.2, worst rooting score: 3.4\n")
            tmp.write("Reconciliation Information\n")
            tmp.write("- Duplications: 5\n")
            tmp.write("- Co-Divergences: 6\n")
            tmp.write("- Transfers: 7\n")
            tmp.write("- Losses: 8\n")
            tmp.write("Tree Without Losses\n")
            tmp.write("x\nx\nx\n")
            tmp.write("- Polytomies: 9\n")
            tmp.write("root positions with rho peak: a b c\n")
            tmp.write("Returning the first MAD tree\n")
        try:
            out_root = kfog.get_notung_root_stats(path)
            self.assertEqual(out_root["ntg_num_opt_root"], 2)
            self.assertAlmostEqual(out_root["ntg_best_root_score"], 1.2)
            self.assertAlmostEqual(out_root["ntg_worst_root_score"], 3.4)

            out_rec = kfog.get_notung_reconcil_stats(path)
            self.assertEqual(out_rec["ntg_num_dup"], 5)
            self.assertEqual(out_rec["ntg_num_codiv"], 6)
            self.assertEqual(out_rec["ntg_num_transfer"], 7)
            self.assertEqual(out_rec["ntg_num_loss"], 8)
            self.assertEqual(out_rec["ntg_num_polytomy"], 9)

            out_rs = kfog.get_root_stats(path)
            self.assertEqual(out_rs["rooting_method"], "MAD")
            self.assertEqual(out_rs["num_rho_peak"], 3)
        finally:
            os.unlink(path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path2 = tmp.name
            tmp.write("  root positions with rho peak: x y z\n")
        try:
            out_rs2 = kfog.get_root_stats(path2)
            self.assertEqual(out_rs2["num_rho_peak"], 3)
        finally:
            os.unlink(path2)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path3 = tmp.name
            tmp.write("INFO: root positions with rho peak:a,b,c\n")
        try:
            out_rs3 = kfog.get_root_stats(path3)
            self.assertEqual(out_rs3["num_rho_peak"], 3)
        finally:
            os.unlink(path3)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path_case = tmp.name
            tmp.write("ROOT POSITIONS WITH RHO PEAK: a b c\n")
            tmp.write("RETURNING THE FIRST MAD tree\n")
        try:
            out_case = kfog.get_root_stats(path_case)
            self.assertEqual(out_case["num_rho_peak"], 3)
            self.assertEqual(out_case["rooting_method"], "MAD")
        finally:
            os.unlink(path_case)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path4 = tmp.name
            tmp.write("root positions with rho peak: -\n")
            tmp.write("root positions with rho peak: none\n")
            tmp.write("root positions with rho peak: NA\n")
        try:
            out_rs4 = kfog.get_root_stats(path4)
            self.assertEqual(out_rs4["num_rho_peak"], 0)
        finally:
            os.unlink(path4)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tsv = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\ttrait2\n")
            tmp.write("n1\talpha\t\t2\t4\n")
            tmp.write("n1\tsigma2\t\t6\t8\n")
            tmp.write("n2\tmu\t1\t1\t2\n")
        try:
            out = kfog.regime2tree(tsv)
            self.assertEqual(out["num_regime"], 2)
            self.assertEqual(out["alpha_trait1"], 2)
            self.assertEqual(out["sigma2_trait2"], 8)
            self.assertAlmostEqual(out["gamma_trait1"], 6 / (2 * 2))
        finally:
            os.unlink(tsv)

        b = pd.DataFrame(
            {
                "orthogroup": ["og1", "og1", "og1", "og2"],
                "branch_id": [0, 1, 2, 0],
                "parent": [1, 2, 2, 0],
                "flag": [0, 1, 0, 1],
                "value": [10, 20, 30, 40],
            }
        )
        self.assertEqual(kfog.get_most_recent(b, 0, "og1", "flag", 1, "value"), 20)
        self.assertTrue(np.isnan(kfog.get_most_recent(b, 0, "og1", "flag", 2, "value")))
        b_dup = pd.DataFrame(
            {
                "orthogroup": ["og1", "og1", "og1", "og1"],
                "branch_id": [0, 0, 1, 2],
                "parent": [1, 2, 2, 2],
                "flag": [1, 0, 0, 0],
                "value": [11, 99, 20, 30],
            }
        )
        self.assertEqual(kfog.get_most_recent(b_dup, 0, "og1", "flag", 1, "value"), 11)

    def test_kfog_regime2tree_accepts_shuffled_trait_columns(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tsv = tmp.name
            tmp.write("trait1\tparam\tregime\tnode_name\ttrait2\n")
            tmp.write("2\talpha\t\tn1\t4\n")
            tmp.write("6\tsigma2\t\tn1\t8\n")
            tmp.write("1\tmu\t1\tn2\t2\n")
        try:
            out = kfog.regime2tree(tsv)
            self.assertEqual(out["num_regime"], 2)
            self.assertEqual(out["alpha_trait1"], 2)
            self.assertEqual(out["sigma2_trait2"], 8)
            self.assertAlmostEqual(out["gamma_trait1"], 6 / (2 * 2))
            self.assertNotIn("alpha_node_name", out)
        finally:
            os.unlink(tsv)

    def test_kfog_get_most_recent_robustness(self):
        b = pd.DataFrame(
            {
                "orthogroup": ["og1", "og1", "og1"],
                "branch_id": [0, 1, 2],
                "parent": [1, 2, 2],
                "flag": [0, 0, 0],
                "value": [10, 20, 30],
            }
        )
        self.assertTrue(np.isnan(kfog.get_most_recent(b, 99, "og1", "flag", 1, "value")))
        self.assertTrue(np.isnan(kfog.get_most_recent(b, 0, "ogX", "flag", 1, "value")))

        b_missing_parent = pd.DataFrame(
            {
                "orthogroup": ["og1", "og1"],
                "branch_id": [0, 2],
                "parent": [1, 2],
                "flag": [0, 0],
                "value": [10, 30],
            }
        )
        self.assertTrue(np.isnan(kfog.get_most_recent(b_missing_parent, 0, "og1", "flag", 1, "value")))

        b_cycle = pd.DataFrame(
            {
                "orthogroup": ["og1", "og1", "og1"],
                "branch_id": [0, 1, 2],
                "parent": [1, 0, 2],
                "flag": [0, 0, 0],
                "value": [10, 20, 30],
            }
        )
        self.assertTrue(np.isnan(kfog.get_most_recent(b_cycle, 0, "og1", "flag", 1, "value")))
        b_non_monotonic = pd.DataFrame(
            {
                "orthogroup": ["og1", "og1", "og1"],
                "branch_id": [0, 99, 10],
                "parent": [99, 10, 10],
                "flag": [0, 0, 1],
                "value": [0, 50, 100],
            }
        )
        self.assertEqual(kfog.get_most_recent(b_non_monotonic, 0, "og1", "flag", 1, "value"), 100)
        with self.assertRaisesRegex(ValueError, "requires columns"):
            kfog.get_most_recent(pd.DataFrame({"orthogroup": ["og1"]}), 0, "og1", "flag", 1, "value")
        with self.assertRaisesRegex(ValueError, "dataframe-like"):
            kfog.get_most_recent(None, 0, "og1", "flag", 1, "value")
        with self.assertRaisesRegex(ValueError, "target_col must be a string"):
            kfog.get_most_recent(b, 0, "og1", ["flag"], 1, "value")
        with self.assertRaisesRegex(ValueError, "return_col must be a string"):
            kfog.get_most_recent(b, 0, "og1", "flag", 1, ["value"])
        with self.assertRaisesRegex(ValueError, "og_col must be a string"):
            kfog.get_most_recent(b, 0, "og1", "flag", 1, "value", og_col=["orthogroup"])
        with self.assertRaisesRegex(ValueError, "nl must be a hashable"):
            kfog.get_most_recent(b, [], "og1", "flag", 1, "value")
        with self.assertRaisesRegex(ValueError, "og must be a hashable"):
            kfog.get_most_recent(b, 0, [], "flag", 1, "value")
        with self.assertRaisesRegex(ValueError, "parent column must contain hashable values"):
            kfog.get_most_recent(
                pd.DataFrame(
                    {
                        "orthogroup": ["og1", "og1"],
                        "branch_id": [0, 1],
                        "parent": [[1], [1]],
                        "flag": [0, 1],
                        "value": [10, 20],
                    }
                ),
                0,
                "og1",
                "flag",
                1,
                "value",
            )
        with self.assertRaisesRegex(ValueError, "branch_id column must contain hashable values"):
            kfog.get_most_recent(
                pd.DataFrame(
                    {
                        "orthogroup": ["og1", "og1"],
                        "branch_id": [[0], [1]],
                        "parent": [1, 1],
                        "flag": [0, 1],
                        "value": [10, 20],
                    }
                ),
                0,
                "og1",
                "flag",
                1,
                "value",
            )
        with self.assertRaisesRegex(ValueError, "branch_id column must not contain missing values"):
            kfog.get_most_recent(
                pd.DataFrame(
                    {
                        "orthogroup": ["og1", "og1"],
                        "branch_id": [0, np.nan],
                        "parent": [1, 1],
                        "flag": [0, 1],
                        "value": [10, 20],
                    }
                ),
                0,
                "og1",
                "flag",
                1,
                "value",
            )
        with self.assertRaisesRegex(ValueError, "unique branch_id"):
            kfog.compute_delta(
                pd.DataFrame({"branch_id": [0, 0], "parent": [1, 1], "x": [1.0, 2.0]}),
                "x",
            )
        with self.assertRaisesRegex(ValueError, "requires columns"):
            kfog.compute_delta(pd.DataFrame({"branch_id": [0], "x": [1.0]}), "x")
        with self.assertRaisesRegex(ValueError, "dataframe-like"):
            kfog.compute_delta(None, "x")
        with self.assertRaisesRegex(ValueError, "column must be a string"):
            kfog.compute_delta(
                pd.DataFrame({"branch_id": [0], "parent": [0], "x": [1.0]}),
                ["x"],
            )
        with self.assertRaisesRegex(ValueError, "requires numeric values"):
            kfog.compute_delta(
                pd.DataFrame({"branch_id": [0, 1], "parent": [1, 1], "x": ["a", "b"]}),
                "x",
            )
        with self.assertRaisesRegex(ValueError, "requires finite numeric values"):
            kfog.compute_delta(
                pd.DataFrame({"branch_id": [0, 1], "parent": [1, 1], "x": [1.0, np.inf]}),
                "x",
            )
        with self.assertRaisesRegex(ValueError, "branch_id column must contain hashable values"):
            kfog.compute_delta(
                pd.DataFrame({"branch_id": [[0], [1]], "parent": [1, 1], "x": [1.0, 2.0]}),
                "x",
            )
        with self.assertRaisesRegex(ValueError, "branch_id column must not contain missing values"):
            kfog.compute_delta(
                pd.DataFrame({"branch_id": [0, np.nan], "parent": [1, 1], "x": [1.0, 2.0]}),
                "x",
            )
        with self.assertRaisesRegex(ValueError, "parent column must contain hashable values"):
            kfog.compute_delta(
                pd.DataFrame({"branch_id": [0, 1], "parent": [[1], [1]], "x": [1.0, 2.0]}),
                "x",
            )

    def test_kfog_notung_parser_robustness(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write("Best rooting score: 1,2, worst rooting score: 3,4\n")
            tmp.write("Best rooting score:1.0,worst rooting score:2.0\n")
            tmp.write("Best rooting score: , worst rooting score: \n")
            tmp.write("Best rooting score: .5, worst rooting score: 1.5\n")
            tmp.write("Best rooting score: -1.2e-3, worst rooting score: +2.3E+2\n")
            tmp.write("Reconciliation Information\n")
            tmp.write("- Duplications: 1,234\n")
            tmp.write("- Co-Divergences: 6\n")
            tmp.write("- Transfers: 7\n")
            tmp.write("- Losses: 8\n")
            tmp.write("root positions with rho peak: a,b,c\n")
        try:
            out_root = kfog.get_notung_root_stats(path)
            self.assertAlmostEqual(out_root["ntg_best_root_score"], -1.2e-3)
            self.assertAlmostEqual(out_root["ntg_worst_root_score"], 2.3e2)
            out_rec = kfog.get_notung_reconcil_stats(path)
            self.assertEqual(out_rec["ntg_num_dup"], 1234)
            self.assertEqual(out_rec["ntg_num_codiv"], 6)
            self.assertEqual(out_rec["ntg_num_transfer"], 7)
            self.assertEqual(out_rec["ntg_num_loss"], 8)
            out_rs = kfog.get_root_stats(path)
            self.assertEqual(out_rs["num_rho_peak"], 3)
        finally:
            os.unlink(path)

    def test_kfog_notung_root_stats_parses_thousands_separators(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write("NUMBER OF OPTIMAL ROOTS: 1,234 OUT OF 5,678\n")
            tmp.write("Best rooting score: 1.234, worst rooting score: 2.345\n")
            tmp.write("Best rooting score: 1,234.5, worst rooting score: 2,345.6\n")
            tmp.write("best rooting score: -1.234,5, worst rooting score: +2.345,6\n")
        try:
            out_root = kfog.get_notung_root_stats(path)
            self.assertEqual(out_root["ntg_num_opt_root"], 1234)
            self.assertAlmostEqual(out_root["ntg_best_root_score"], -1234.5)
            self.assertAlmostEqual(out_root["ntg_worst_root_score"], 2345.6)
        finally:
            os.unlink(path)

    def test_kfog_notung_reconcil_stats_parses_flexible_order_and_spacing(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write("Reconciliation Information\n")
            tmp.write("- Losses:8\n")
            tmp.write("- Transfers:7\n")
            tmp.write("- Co-Divergences:6\n")
            tmp.write("- Duplications:1,234\n")
            tmp.write("Tree Without Losses\n")
            tmp.write("x\nx\nx\n")
            tmp.write("- Polytomies:9\n")
        try:
            out_rec = kfog.get_notung_reconcil_stats(path)
            self.assertEqual(out_rec["ntg_num_dup"], 1234)
            self.assertEqual(out_rec["ntg_num_codiv"], 6)
            self.assertEqual(out_rec["ntg_num_transfer"], 7)
            self.assertEqual(out_rec["ntg_num_loss"], 8)
            self.assertEqual(out_rec["ntg_num_polytomy"], 9)
        finally:
            os.unlink(path)

    def test_kfog_regime2tree_input_validation(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            empty_data_path = tmp.name
            tmp.write("param\tregime\n")
        try:
            with self.assertRaisesRegex(ValueError, "at least one data row"):
                kfog.regime2tree(empty_data_path)
        finally:
            os.unlink(empty_data_path)
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            bad_utf8_path = tmp.name
            tmp.write(b"\xff\xfe\xfd")
        try:
            with self.assertRaisesRegex(ValueError, "UTF-8 tab-separated text"):
                kfog.regime2tree(bad_utf8_path)
        finally:
            os.unlink(bad_utf8_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            empty_file_path = tmp.name
            tmp.write("")
        try:
            with self.assertRaisesRegex(ValueError, "UTF-8 tab-separated text"):
                kfog.regime2tree(empty_file_path)
        finally:
            os.unlink(empty_file_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            missing_col_path = tmp.name
            tmp.write("node_name\tparam\ttrait1\n")
            tmp.write("n1\talpha\t2\n")
        try:
            with self.assertRaisesRegex(ValueError, "requires columns"):
                kfog.regime2tree(missing_col_path)
        finally:
            os.unlink(missing_col_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            no_trait_path = tmp.name
            tmp.write("node_name\tparam\tregime\n")
            tmp.write("n1\talpha\t\n")
        try:
            with self.assertRaisesRegex(ValueError, "at least one trait column"):
                kfog.regime2tree(no_trait_path)
        finally:
            os.unlink(no_trait_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            invalid_regime_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\tx\t2\n")
        try:
            with self.assertRaisesRegex(ValueError, "must be numeric or NaN"):
                kfog.regime2tree(invalid_regime_path)
        finally:
            os.unlink(invalid_regime_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            negative_regime_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\t-1\t2\n")
        try:
            with self.assertRaisesRegex(ValueError, "non-negative IDs"):
                kfog.regime2tree(negative_regime_path)
        finally:
            os.unlink(negative_regime_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            non_integer_regime_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\t1.5\t2\n")
        try:
            with self.assertRaisesRegex(ValueError, "integer IDs"):
                kfog.regime2tree(non_integer_regime_path)
        finally:
            os.unlink(non_integer_regime_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            inf_regime_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\tinf\t2\n")
        try:
            with self.assertRaisesRegex(ValueError, "finite numeric values"):
                kfog.regime2tree(inf_regime_path)
        finally:
            os.unlink(inf_regime_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            huge_regime_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\t9223372036854775808\t2\n")
        try:
            with self.assertRaisesRegex(ValueError, "avoid integer overflow"):
                kfog.regime2tree(huge_regime_path)
        finally:
            os.unlink(huge_regime_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            zero_alpha_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\t\t0\n")
            tmp.write("n1\tsigma2\t\t1\n")
        try:
            with self.assertRaisesRegex(ValueError, "must be non-zero"):
                kfog.regime2tree(zero_alpha_path)
        finally:
            os.unlink(zero_alpha_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            conflicting_param_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\t\t2\n")
            tmp.write("n2\talpha\t\t3\n")
        try:
            with self.assertRaisesRegex(ValueError, "conflicting values for param"):
                kfog.regime2tree(conflicting_param_path)
        finally:
            os.unlink(conflicting_param_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            missing_param_name_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\t\t\t2\n")
        try:
            with self.assertRaisesRegex(ValueError, "non-empty param names"):
                kfog.regime2tree(missing_param_name_path)
        finally:
            os.unlink(missing_param_name_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            all_nan_regime_path = tmp.name
            tmp.write("node_name\tparam\tregime\ttrait1\n")
            tmp.write("n1\talpha\t\t2\n")
        try:
            out = kfog.regime2tree(all_nan_regime_path)
            self.assertEqual(out["num_regime"], 0)
        finally:
            os.unlink(all_nan_regime_path)

    def test_kfog_ou2table_input_validation(self):
        with self.assertRaisesRegex(ValueError, "input_tree_file must be an existing file path"):
            kfog.ou2table(
                "/tmp/definitely_missing_kftools_regime_abc.tsv",
                "/tmp/definitely_missing_kftools_leaf_abc.tsv",
                "/tmp/definitely_missing_kftools_tree_abc.nwk",
            )

        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path_missing_regime = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tt1\n")
            leaf_tmp.write("x\tmu\t1.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "leaf_file requires columns"):
                kfog.ou2table(regime_path, leaf_path_missing_regime, tree_path)
        finally:
            os.unlink(regime_path)
            os.unlink(leaf_path_missing_regime)
            os.unlink(tree_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            regime_non_integer_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t1.5\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "must contain integer IDs"):
                kfog.ou2table(regime_non_integer_path, leaf_path, tree_path)
        finally:
            os.unlink(regime_non_integer_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            regime_inf_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\tinf\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "finite numeric values"):
                kfog.ou2table(regime_inf_path, leaf_path, tree_path)
        finally:
            os.unlink(regime_inf_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            regime_empty_path = regime_tmp.name
            regime_tmp.write("")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("(A_x:1,B_x:1)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "UTF-8 tab-separated text"):
                kfog.ou2table(regime_empty_path, leaf_path, tree_path)
        finally:
            os.unlink(regime_empty_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            conflicting_regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t1\n")
            regime_tmp.write("N1\t2\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\n")
            leaf_tmp.write("x\tmu\t2\t3.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "conflicting regime IDs"):
                kfog.ou2table(conflicting_regime_path, leaf_path, tree_path)
        finally:
            os.unlink(conflicting_regime_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            unknown_node_name_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("UnknownNode\t1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "not present in input_tree_file"):
                kfog.ou2table(unknown_node_name_path, leaf_path, tree_path)
        finally:
            os.unlink(unknown_node_name_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            duplicate_tree_name_regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,(C_x:1,D_x:1)N1:1)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "duplicate non-empty node names"):
                kfog.ou2table(duplicate_tree_name_regime_path, leaf_path, tree_path)
        finally:
            os.unlink(duplicate_tree_name_regime_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)

        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            negative_regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t-1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "non-negative IDs"):
                kfog.ou2table(negative_regime_path, leaf_path, tree_path)
        finally:
            os.unlink(negative_regime_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            huge_regime_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("N1\t9223372036854775808\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t9223372036854775808\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "avoid integer overflow"):
                kfog.ou2table(huge_regime_path, leaf_path, tree_path)
        finally:
            os.unlink(huge_regime_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as regime_tmp:
            missing_node_name_path = regime_tmp.name
            regime_tmp.write("node_name\tregime\n")
            regime_tmp.write("\t1\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as leaf_tmp:
            leaf_path = leaf_tmp.name
            leaf_tmp.write("node_name\tparam\tregime\tt1\n")
            leaf_tmp.write("x\tmu\t0\t1.0\n")
            leaf_tmp.write("x\tmu\t1\t2.0\n")
        with tempfile.NamedTemporaryFile("w", delete=False) as tree_tmp:
            tree_path = tree_tmp.name
            tree_tmp.write("((A_x:1,B_x:1)N1:1,C_x:2)Root;\n")
        try:
            with self.assertRaisesRegex(ValueError, "node_name column must contain non-empty string values"):
                kfog.ou2table(missing_node_name_path, leaf_path, tree_path)
        finally:
            os.unlink(missing_node_name_path)
            os.unlink(leaf_path)
            os.unlink(tree_path)

    def test_kfplot(self):
        df = pd.DataFrame(
            {"v": np.random.normal(size=80), "g": ["A"] * 40 + ["B"] * 40}
        )
        ax = kfplot.hist_boxplot(x="v", category="g", df=df, xlim=[-3, 3])
        self.assertIsNotNone(ax)
        import matplotlib.pyplot as plt
        df_bar = pd.DataFrame({"x1": [1, 2], "x2": [2, 3], "y": ["A", "B"]})
        fig2, ax2 = plt.subplots()
        out_ax2 = kfplot.stacked_barplot(x=["x1", "x2"], y="y", data=df_bar, colors=["C0", "C1"], ax=ax2)
        self.assertIsNotNone(out_ax2)
        colors = ["C0", "C1"]
        fig3, ax3 = plt.subplots()
        _ = kfplot.hist_boxplot(x="v", category="g", df=df, colors=colors, xlim=[-3, 3], ax=ax3)
        self.assertEqual(colors, ["C0", "C1"])
        plt.close(fig3)
        fig3_tuple, ax3_tuple = plt.subplots()
        df_single_cat = pd.DataFrame({"v": [1.0, 2.0, 3.0], "g": ["A", "A", "A"]})
        _ = kfplot.hist_boxplot(
            x="v",
            category="g",
            df=df_single_cat,
            colors=("red",),
            xlim=[0, 4],
            ax=ax3_tuple,
        )
        self.assertEqual(ax3_tuple.lines[0].get_color(), "red")
        plt.close(fig3_tuple)
        fig3b, ax3b = plt.subplots()
        _ = kfplot.hist_boxplot(
            x="v",
            category="g",
            df=df,
            colors={"A": "C0"},
            xlim=[-3, 3],
            ax=ax3b,
        )
        yticklabels = [tick.get_text() for tick in ax3b.get_yticklabels()]
        self.assertIn("A", yticklabels)
        self.assertIn("B", yticklabels)
        plt.close(fig3b)
        fig4, ax4 = plt.subplots()
        out_ax4 = kfplot.density_scatter(
            x=df["v"].values,
            y=(df["v"].values * 0.5) + np.random.normal(scale=0.1, size=df.shape[0]),
            ax=ax4,
            cbar=False,
            show_cor_p=True,
        )
        self.assertIsNotNone(out_ax4)
        plt.close(fig4)
        plt.close(fig2)
        plt.close(ax.figure)

    def test_kfplot_density_scatter_empty(self):
        import statsmodels.api as sm
        with self.assertRaises(ValueError):
            kfplot.density_scatter(x=[np.nan], y=[np.nan], cbar=False)
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfplot.density_scatter(x=[1, {}], y=[1, 2], cbar=False)
        with self.assertRaisesRegex(ValueError, "same shape"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2, 3], cbar=False)
        with self.assertRaisesRegex(ValueError, "must include columns"):
            kfplot.density_scatter(x="x", y="y", df=pd.DataFrame({"x": [1, 2]}), cbar=False)
        with self.assertRaisesRegex(ValueError, "DataFrame-like"):
            kfplot.density_scatter(x="x", y="y", df=[], cbar=False)
        with self.assertRaisesRegex(ValueError, "string column name"):
            kfplot.density_scatter(x=["x"], y="y", df=pd.DataFrame({"x": [1, 2], "y": [1, 2]}), cbar=False)
        with self.assertRaisesRegex(ValueError, "string column name"):
            kfplot.density_scatter(x="x", y=["y"], df=pd.DataFrame({"x": [1, 2], "y": [1, 2]}), cbar=False)
        with self.assertRaisesRegex(ValueError, "must contain numeric values"):
            kfplot.density_scatter(
                x="x",
                y="y",
                df=pd.DataFrame({"x": ["a", "b"], "y": [1, 2]}),
                cbar=False,
            )
        with self.assertRaisesRegex(ValueError, "cor must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], cor="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "diag must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], diag="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "hue_log must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], hue_log="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "show_cor_p must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], show_cor_p="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "return_ims must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], return_ims="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "cbar must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], cbar="False")
        ax = kfplot.density_scatter(
            x="x",
            y="y",
            df=pd.DataFrame({"x": ["1.0", "inf"], "y": [1.0, 2.0]}),
            cbar=False,
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close(ax.figure)
        ax_same_col = kfplot.density_scatter(
            x="x",
            y="x",
            df=pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            cbar=False,
        )
        self.assertIsNotNone(ax_same_col)
        matplotlib.pyplot.close(ax_same_col.figure)
        with self.assertRaisesRegex(ValueError, "num_bin must be a positive integer"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], num_bin=0, cbar=False)
        with self.assertRaisesRegex(ValueError, "num_bin must be a positive integer"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], num_bin=2.5, cbar=False)
        with self.assertRaisesRegex(ValueError, "vmin must be None or a finite numeric value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], vmin=np.nan, cbar=False)
        with self.assertRaisesRegex(ValueError, "vmax must be None or a finite numeric value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], vmax=np.inf, cbar=False)
        with self.assertRaisesRegex(ValueError, "vmin must be less than or equal to vmax"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], vmin=2, vmax=1, cbar=False)
        with self.assertRaisesRegex(ValueError, "plot_range must"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range=[0, 1], cbar=False)
        with self.assertRaisesRegex(ValueError, "plot_range must"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range="bad", cbar=False)
        with self.assertRaisesRegex(ValueError, "xmin <= xmax"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range=[2, 1, 0, 1], cbar=False)
        with self.assertRaisesRegex(ValueError, "ymin <= ymax"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range=[0, 1, 2, 1], cbar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "GLM fit failed"):
                kfplot.density_scatter(x=[1, 2, 3], y=[0, 0, 0], reg_family=sm.families.Poisson(), cbar=False)
        with self.assertRaisesRegex(ValueError, "reg_family must be a statsmodels family object"):
            kfplot.density_scatter(x=[1, 2, 3], y=[1, 2, 3], reg_family="bad", cbar=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(
                x=[1, 2, 3, 4],
                y=[1, 2, 3, 4],
                reg_family=sm.families.Poisson(),
                cbar=False,
            )
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("log link function was detected" in str(wi.message) for wi in w),
                "density_scatter should not emit runtime warnings for expected Poisson log-link handling",
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(
                x=[1, 2, 3, 4],
                y=[0, 1, 2, 3],
                reg_family=sm.families.Poisson(),
                cbar=False,
            )
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("divide by zero encountered in log" in str(wi.message) for wi in w),
                "density_scatter should suppress NumPy divide warnings during log-link transformation",
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(x=[1.0], y=[1.0], cbar=False)
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("identical low and high" in str(wi.message) for wi in w),
                "density_scatter should avoid singular-axis matplotlib warnings for single-point inputs",
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(x=[1.0], y=[1.0], cbar=False, show_cor_p=False)
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("Degrees of freedom <= 0" in str(wi.message) for wi in w),
                "density_scatter(show_cor_p=False) should avoid corrcoef runtime warnings for single-point inputs",
            )

    def test_kfplot_stacked_barplot_input_validation(self):
        import matplotlib.pyplot as plt
        df = pd.DataFrame({"a": [1], "b": [2]})
        df_bad_x = pd.DataFrame({"a": [1, 2], "c": ["x", "y"], "g": ["A", "B"]})
        df_bad_y = pd.DataFrame({"g": ["A", "B"], "y1": [1, 2], "y2": ["x", "y"]})
        df_bad_group = pd.DataFrame({"a": [1, 2], "g": [[1], [2]]})
        fig, ax = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "Exactly one of x and y"):
                kfplot.stacked_barplot(x="a", y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "DataFrame-like"):
                kfplot.stacked_barplot(x=["a"], y="b", data=None, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "must contain at least one"):
                kfplot.stacked_barplot(x=[], y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x column name must be a non-empty"):
                kfplot.stacked_barplot(x="", y=["y1"], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y column name must be a non-empty"):
                kfplot.stacked_barplot(x=["a"], y="", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x list must contain non-empty string"):
                kfplot.stacked_barplot(x=[{}], y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y list must contain non-empty string"):
                kfplot.stacked_barplot(x="g", y=[None], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x list must not contain duplicate"):
                kfplot.stacked_barplot(x=["a", "a"], y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y list must not contain duplicate"):
                kfplot.stacked_barplot(x="g", y=["y1", "y1"], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x columns must contain numeric values"):
                kfplot.stacked_barplot(x=["a", "c"], y="g", data=df_bad_x, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y columns must contain numeric values"):
                kfplot.stacked_barplot(x="g", y=["y1", "y2"], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "must contain hashable values"):
                kfplot.stacked_barplot(x=["a"], y="g", data=df_bad_group, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "must contain scalar values"):
                kfplot.stacked_barplot(
                    x=["a"],
                    y="g",
                    data=pd.DataFrame({"a": [1, 2], "g": [("A",), ("B",)]}),
                    colors=["C0"],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must not contain complex values"):
                kfplot.stacked_barplot(
                    x="g",
                    y=["a"],
                    data=pd.DataFrame({"g": [1 + 2j, 3 + 4j], "a": [1, 2]}),
                    colors=["C0"],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must not mix bool and numeric non-bool values"):
                kfplot.stacked_barplot(
                    x=["a"],
                    y="g",
                    data=pd.DataFrame({"a": [1, 2], "g": [1, True]}),
                    colors=["C0"],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must not contain non-finite numeric values"):
                kfplot.stacked_barplot(
                    x="x",
                    y=["a"],
                    data=pd.DataFrame({"x": [0.1, np.inf], "a": [1, 2]}),
                    colors=["C0"],
                    ax=ax,
                )
            out_bool = kfplot.stacked_barplot(
                x="g",
                y=["y1", "y2"],
                data=pd.DataFrame({"g": [1], "y1": [True], "y2": [1]}),
                colors=["C0", "C1"],
                ax=None,
            )
            self.assertIsNotNone(out_bool)
            plt.close(out_bool.figure)
            out_datetime = kfplot.stacked_barplot(
                x="g",
                y=["y1", "y2"],
                data=pd.DataFrame(
                    {
                        "g": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                        "y1": [1, 2],
                        "y2": [2, 3],
                    }
                ),
                colors=["C0", "C1"],
                ax=None,
            )
            self.assertIsNotNone(out_datetime)
            plt.close(out_datetime.figure)
            out_numpy_scalar_color = kfplot.stacked_barplot(
                x="g",
                y=["y1", "y2"],
                data=pd.DataFrame({"g": ["A", "B"], "y1": [1, 2], "y2": [2, 3]}),
                colors=np.array("red"),
                ax=None,
            )
            self.assertIsNotNone(out_numpy_scalar_color)
            plt.close(out_numpy_scalar_color.figure)
            with self.assertRaisesRegex(ValueError, "must not mix string and non-string"):
                fig_count_before = len(plt.get_fignums())
                kfplot.stacked_barplot(
                    x="g",
                    y=["y1", "y2"],
                    data=pd.DataFrame({"g": [1, "2"], "y1": [1, 2], "y2": [2, 3]}),
                    colors=["C0", "C1"],
                    ax=None,
                )
            self.assertEqual(
                len(plt.get_fignums()),
                fig_count_before,
                "stacked_barplot should close internally created figures when validation fails after figure creation",
            )
            out = kfplot.stacked_barplot(x=["a"], y="b", data=df, colors=["C0"], ax=None)
            self.assertIsNotNone(out)
            plt.close(out.figure)
        finally:
            plt.close(fig)

    def test_kfplot_empty_input_validation(self):
        with self.assertRaisesRegex(ValueError, "DataFrame-like"):
            kfplot.hist_boxplot(x="x", category="g", df=[])
        with self.assertRaisesRegex(ValueError, "x must be a non-empty string"):
            kfplot.hist_boxplot(x=["x"], category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "category must be a non-empty string"):
            kfplot.hist_boxplot(x="x", category=["g"], df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "x must be a non-empty string"):
            kfplot.hist_boxplot(x="", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "category must be a non-empty string"):
            kfplot.hist_boxplot(x="x", category="", df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "alpha must be a finite numeric value"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), alpha="bad")
        with self.assertRaisesRegex(ValueError, "alpha must be between 0 and 1"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), alpha=2)
        with self.assertRaisesRegex(ValueError, "box_step must be a positive finite numeric value"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), box_step="bad")
        with self.assertRaisesRegex(ValueError, "box_step must be a positive finite numeric value"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), box_step=-0.1)
        with self.assertRaisesRegex(ValueError, "must contain hashable values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [[1], [2]]}),
            )
        with self.assertRaisesRegex(ValueError, "must contain scalar values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [("A",), ("B",)]}),
            )
        with self.assertRaisesRegex(ValueError, "must not contain complex values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [1 + 2j, 3 + 4j]}),
            )
        with self.assertRaisesRegex(ValueError, "must not mix bool and numeric non-bool values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [1, True]}),
            )
        with self.assertRaisesRegex(ValueError, "must not contain non-finite numeric values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [np.inf, 1]}),
            )
        with self.assertRaisesRegex(ValueError, "at least one non-NaN"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame(columns=["x", "g"]))
        with self.assertRaisesRegex(ValueError, "category column"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0]}))
        with self.assertRaisesRegex(ValueError, "x column"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "at least one non-NaN"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [np.nan, np.nan]}),
            )
        with self.assertRaisesRegex(ValueError, "xlim must"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                xlim=[0],
            )
        with self.assertRaisesRegex(ValueError, "xmin <= xmax"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                xlim=[2, 1],
            )
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": ["bad", "2"], "g": ["A", "B"]}),
            )
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, np.inf], "g": ["A", "B"]}),
            )
        with self.assertRaisesRegex(ValueError, "bins must"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins="bad",
            )
        with self.assertRaisesRegex(ValueError, "finite numeric bin-edge values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=["a", "b"],
            )
        with self.assertRaisesRegex(ValueError, "at least 2 bin-edge"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=[1],
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=[2, 1],
            )
        with self.assertRaisesRegex(ValueError, "finite numeric bin-edge values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=[1, np.inf],
            )
        with self.assertRaisesRegex(ValueError, "colors contains categories"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                colors={"A": "C0", "Z": "C1"},
            )
        import matplotlib.pyplot as plt
        fig_count_before = len(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "has no non-NaN values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, np.nan], "g": ["A", "B"]}),
            )
        self.assertEqual(
            len(plt.get_fignums()),
            fig_count_before,
            "hist_boxplot should close internally created figures when validation fails after figure creation",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax_bool_x = kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [True, False, True], "g": ["A", "A", "B"]}),
            )
            self.assertIsNotNone(ax_bool_x)
            self.assertFalse(
                any("Converting input from bool" in str(wi.message) for wi in w),
                "hist_boxplot should coerce boolean x values to float and avoid matplotlib bool histogram warnings",
            )
            plt.close(ax_bool_x.figure)
        ax_datetime_category = kfplot.hist_boxplot(
            x="x",
            category="g",
            df=pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "g": [
                        pd.Timestamp("2020-01-01"),
                        pd.Timestamp("2020-01-01"),
                        pd.Timestamp("2020-01-02"),
                    ],
                }
            ),
        )
        self.assertIsNotNone(ax_datetime_category)
        plt.close(ax_datetime_category.figure)
        ax_numpy_scalar_color = kfplot.hist_boxplot(
            x="x",
            category="g",
            df=pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["A", "A", "B"]}),
            colors=np.array("red"),
        )
        self.assertIsNotNone(ax_numpy_scalar_color)
        plt.close(ax_numpy_scalar_color.figure)
        fig, ax = plt.subplots()
        try:
            out = kfplot.ols_annotations(
                x=[1, 2, 3],
                y=[2, 3, 4],
                ax=ax,
                method="ols",
            )
            self.assertIsNotNone(out)
            with self.assertRaisesRegex(ValueError, "at least 2 rows"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame(columns=["x", "y"]),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "DataFrame-like"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=[],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "string column name"):
                kfplot.ols_annotations(
                    x=["x"],
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "string column name"):
                kfplot.ols_annotations(
                    x="x",
                    y=["y"],
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must include columns"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must be either"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    method="bad",
                )
            with self.assertRaisesRegex(ValueError, "requires numeric x and y"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": ["a", "b"], "y": [3, 4]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "stats must be a string or a sequence"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    stats=1,
                )
            with self.assertRaisesRegex(ValueError, "unsupported entries"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    stats=["N", "bad"],
                )
            with self.assertRaisesRegex(ValueError, "unsupported entries"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    stats=["N", 1],
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy=[],
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy=[0.1],
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy="bad",
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy=None,
                )
            out_stats_str = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                ax=ax,
                stats="N",
            )
            self.assertIsNotNone(out_stats_str)
            indexed_df = pd.DataFrame(
                {"x": [3.0, 1.0, 2.0], "y": [2.9, 1.1, 2.1]},
                index=[10, 30, 20],
            )
            out_idx = kfplot.ols_annotations(
                x="x",
                y="y",
                data=indexed_df,
                ax=ax,
                method="ols",
            )
            self.assertIsNotNone(out_idx)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_two_point = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [2, False], "y": [1, 0]}),
                    ax=ax,
                    method="ols",
                    stats=["N", "rsquared", "rsquared_p"],
                )
                self.assertIsNotNone(out_two_point)
                self.assertFalse(
                    any("divide by zero encountered in divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid rsquared divide-by-zero warnings when residual dof is 0",
                )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_quantreg_degenerate = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [0, 0], "y": [0, False]}),
                    ax=ax,
                    method="quantreg",
                    stats=["N", "rsquared", "rsquared_p"],
                )
                self.assertIsNotNone(out_quantreg_degenerate)
                self.assertFalse(
                    any("invalid value encountered in scalar divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid quantreg prsquared divide warnings for degenerate inputs",
                )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_quantreg_constant_y = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame(
                        {
                            "x": [
                                -0.11544881657289713,
                                -0.9731576166416964,
                                1.5521497178256616,
                                0.8980692396695097,
                                0.8121127598698327,
                                -0.15301988744981074,
                                1.995012141280733,
                                0.2387915074956506,
                                0.02822408070062835,
                                1.2000253792148359,
                                -0.5006654582078435,
                                0.5738987504794617,
                                -1.0060237130999417,
                                1.23486392210854,
                            ],
                            "y": [0.0] * 14,
                        }
                    ),
                    ax=ax,
                    method="quantreg",
                    stats=["N", "slope", "slope_p"],
                )
                self.assertIsNotNone(out_quantreg_constant_y)
                self.assertFalse(
                    any("divide by zero encountered in scalar divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid quantreg fit divide-by-zero warnings when y has no variation",
                )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_ols_constant_y = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [2, False, True], "y": [1, True, 1]}),
                    ax=ax,
                    method="ols",
                    stats=["N", "rsquared", "rsquared_p"],
                )
                self.assertIsNotNone(out_ols_constant_y)
                self.assertFalse(
                    any("divide by zero encountered in scalar divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid OLS rsquared divide warnings when y has no variation",
                )
            out_quantreg_missing_slope = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [False, True], "y": [0, 1]}),
                ax=ax,
                method="quantreg",
                stats=["N", "rsquared_p"],
            )
            self.assertIsNotNone(out_quantreg_missing_slope)
            out_quantreg_missing_slope_requested = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [False, True], "y": [0, 1]}),
                ax=ax,
                method="quantreg",
                stats=["slope", "slope_p"],
            )
            self.assertIsNotNone(out_quantreg_missing_slope_requested)
            out_bool_y = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [3, 3], "y": [False, False]}),
                ax=ax,
                method="ols",
                stats=["slope", "slope_p"],
            )
            self.assertIsNotNone(out_bool_y)
        finally:
            plt.close(fig)

    def test_kfstat(self):
        x = np.random.normal(size=200)
        y = np.random.normal(loc=0.2, size=200)
        out = kfstat.bm_test(x, y)
        self.assertEqual(len(out), 6)

    def test_kfstat_input_validation(self):
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.bm_test([1], [2, 3])
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.brunner_munzel_test([1], [2])
        with self.assertRaisesRegex(ValueError, "ttype must be a finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], ttype="bad")
        with self.assertRaisesRegex(ValueError, "ttype must be a finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], ttype=np.nan)
        with self.assertRaisesRegex(ValueError, "ttype must be a finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], ttype=np.inf)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfstat.bm_test([1, 2], [2, 3], alpha=0)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfstat.bm_test([1, 2], [2, 3], alpha=1)
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], alpha=np.nan)
        with self.assertRaisesRegex(ValueError, "one of"):
            kfstat.brunner_munzel_test([1, 2], [2, 3], alternative="bad")
        out_two_sided_dash = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="two-sided")
        out_two_sided_dot = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="two.sided")
        out_two_sided_space = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="two sided")
        out_two_sided_upper = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="TWO_SIDED")
        self.assertAlmostEqual(out_two_sided_dash[1], out_two_sided_dot[1])
        self.assertAlmostEqual(out_two_sided_dash[1], out_two_sided_space[1])
        self.assertAlmostEqual(out_two_sided_dash[1], out_two_sided_upper[1])
        with self.assertRaisesRegex(ValueError, "must be a string"):
            kfstat.brunner_munzel_test([1, 2], [2, 3], alternative=None)
        with self.assertRaisesRegex(ValueError, "must contain numeric values"):
            kfstat.bm_test([1, {}], [2, 3])
        with self.assertRaisesRegex(ValueError, "must contain numeric values"):
            kfstat.brunner_munzel_test([1, {}], [2, 3])
        x_with_nan = [0.0, 1.0, np.nan, 2.0]
        y_with_inf = [1.0, 2.0, np.inf, 3.0]
        stat_filtered, p_filtered = kfstat.brunner_munzel_test(x_with_nan, y_with_inf, alternative="two_sided")
        stat_ref = stats.brunnermunzel([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], alternative="two-sided").statistic
        p_ref = stats.brunnermunzel([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], alternative="two-sided").pvalue
        self.assertAlmostEqual(stat_filtered, stat_ref)
        self.assertAlmostEqual(p_filtered, p_ref)
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.brunner_munzel_test([np.nan, np.inf], [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.brunner_munzel_test([1.0, 2.0, 3.0], [np.nan, np.inf])
        with self.assertRaisesRegex(ValueError, "pooled variance is zero"):
            kfstat.bm_test([1, 1], [2, 2])
        with self.assertRaisesRegex(ValueError, "pooled variance is zero"):
            kfstat.brunner_munzel_test([1, 1], [2, 2])
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, 200)
        y = rng.normal(1.0, 1.0, 200)
        bm_two_sided = kfstat.bm_test(x, y)[2]
        bm_less = kfstat.bm_test(x, y, ttype=1)[2]
        bm_greater = kfstat.bm_test(x, y, ttype=-1)[2]
        bm_two_sided_ref = stats.brunnermunzel(x, y, alternative="two-sided").pvalue
        bm_less_ref = stats.brunnermunzel(x, y, alternative="less").pvalue
        bm_greater_ref = stats.brunnermunzel(x, y, alternative="greater").pvalue
        self.assertGreater(bm_two_sided, 0.0)
        self.assertGreater(bm_less, 0.0)
        self.assertAlmostEqual(bm_two_sided, bm_two_sided_ref)
        self.assertAlmostEqual(bm_less, bm_less_ref)
        self.assertAlmostEqual(bm_greater, bm_greater_ref)
        bm2_two_sided = kfstat.brunner_munzel_test(x, y, alternative="two_sided")[1]
        bm2_less = kfstat.brunner_munzel_test(x, y, alternative="less")[1]
        self.assertGreater(bm2_two_sided, 0.0)
        self.assertGreater(bm2_less, 0.0)
        self.assertAlmostEqual(bm2_two_sided, bm_two_sided_ref)
        self.assertAlmostEqual(bm2_less, bm_less_ref)


if __name__ == "__main__":
    unittest.main()
