import gzip
import io
import tempfile
import unittest
import contextlib
from pathlib import Path

import ete4
import matplotlib
import numpy
import pandas

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
        self.assertEqual(kfutil.rgb_to_hex(1, 0.5, 0), "#FF8000")
        self.assertEqual(len(kfutil.get_rgb_gradient(5, [1, 0, 0], [0, 0, 1])), 5)

    def test_kfexpression(self):
        df = pandas.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
        tau = kfexpression.calc_tau(df, ["a", "b"], unlog2=False, unPlus1=False)
        self.assertEqual(len(tau), 2)
        self.assertTrue(numpy.isfinite(tau).all())
        self.assertAlmostEqual(kfexpression.calc_complementarity([1, 2], [1, 1]), 0.25)
        self.assertAlmostEqual(kfexpression.calc_complementarity([1, 2, 3], [1]), 0.0)

    def test_kfphylo(self):
        tree = ete4.PhyloTree("((A:1,B:1):2,C:3);", parser=1)
        out = kfphylo.add_numerical_node_labels(tree)
        labels = [node.branch_id for node in out.traverse()]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(kfphylo.check_ultrametric(tree))

    def test_kfphylo_load_phylo_tree(self):
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
            import os
            os.unlink(tree_path)
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            empty_tree_path = Path(tmp.name)
        try:
            with self.assertRaisesRegex(ValueError, "empty"):
                kfphylo.load_phylo_tree(empty_tree_path, parser=1)
        finally:
            import os
            os.unlink(empty_tree_path)
        with self.assertRaises(ValueError):
            kfphylo.load_phylo_tree(None, parser=1)
        with self.assertRaises(ValueError):
            kfphylo.load_phylo_tree("   ", parser=1)
        with self.assertRaises(TypeError):
            kfphylo.load_phylo_tree(123, parser=1)
        with self.assertRaisesRegex(ValueError, "not a file"):
            kfphylo.load_phylo_tree(Path("."), parser=1)

    def test_kfphylo_transfer_root(self):
        tree_from = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to = ete4.PhyloTree("(A:1,(B:1,(C:1,D:1):2):2);", parser=1)
        out = kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)
        self.assertEqual(set(out.leaf_names()), set(tree_from.leaf_names()))
        self.assertEqual(len(out.get_children()), 2)

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
        with self.assertRaisesRegex(ValueError, "root split"):
            kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)

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

    def test_kfphylo_taxonomic_annotation_validates_leaf_names(self):
        tree = ete4.PhyloTree("(A:1,B_c:1);", parser=1)
        with self.assertRaisesRegex(ValueError, "genus and species"):
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

    def test_kfog(self):
        newick = "((A_a:1,B_b:1):1,C_c:2);"
        df = kfog.nwk2table(newick, attr="dist", age=True)
        self.assertGreater(len(df), 0)
        df2 = kfog.nwk2table(newick, attr="dist", age=False, parent=True, sister=True)
        self.assertIn("parent", df2.columns)
        self.assertIn("sister", df2.columns)
        self.assertEqual(df2["branch_id"].tolist(), sorted(df2["branch_id"].tolist()))

    def test_kfog_nwk2table_age_requires_ultrametric(self):
        non_ultrametric = "((A_a:1,B_b:2):1,C_c:2);"
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "ultrametric"):
                kfog.nwk2table(non_ultrametric, attr="dist", age=True)

    def test_kfog_nwk2table_pathlike_input(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tree_path = Path(tmp.name)
            tmp.write("((A_a:1,B_b:1):1,C_c:2);")
        try:
            df = kfog.nwk2table(tree_path, attr="dist")
            self.assertGreater(len(df), 0)
        finally:
            import os
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
            import os
            os.unlink(path)

    def test_kfog_iqtree_stats(self):
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            path = tmp.name
        try:
            with gzip.open(path, "wb") as f:
                f.write(b"best_model_AIC: M1\n")
            out = kfog.get_iqtree_model_stats(path)
            self.assertEqual(out["iqtree_best_AIC"], "M1")
        finally:
            import os
            os.unlink(path)

    def test_kfog_node_gene2species_ultrametric(self):
        species_tree = ete4.PhyloTree("((A_x:1,B_x:1):1,(C_x:1,D_x:1):1);", parser=1)
        gene_tree = ete4.PhyloTree("((A_x_g1:1,B_x_g2:1):1,(C_x_g3:1,D_x_g4:1):1);", parser=1)
        out = kfog.node_gene2species(gene_tree, species_tree, is_ultrametric=True)
        self.assertIn("spnode_coverage", out.columns)
        self.assertIn("spnode_age", out.columns)
        self.assertEqual(len(out), len(list(gene_tree.traverse())))

    def test_kfog_node_gene2species_validates_gene_leaf_name(self):
        species_tree = "((A_x:1,B_x:1):1,C_x:2);"
        gene_tree = "((A_x_g1:1,Bx:1):1,C_x_g3:2);"
        with self.assertRaisesRegex(ValueError, "Gene leaf name"):
            kfog.node_gene2species(gene_tree, species_tree, is_ultrametric=False)

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
            import os
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
            import os
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
            import os
            os.unlink(path)

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
            self.assertEqual(out_rs["num_rho_peak"], 2)
        finally:
            import os
            os.unlink(path)

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
            import os
            os.unlink(tsv)

        b = pandas.DataFrame(
            {
                "orthogroup": ["og1", "og1", "og1", "og2"],
                "branch_id": [0, 1, 2, 0],
                "parent": [1, 2, 2, 0],
                "flag": [0, 1, 0, 1],
                "value": [10, 20, 30, 40],
            }
        )
        self.assertEqual(kfog.get_most_recent(b, 0, "og1", "flag", 1, "value"), 20)
        self.assertTrue(numpy.isnan(kfog.get_most_recent(b, 0, "og1", "flag", 2, "value")))
        b_dup = pandas.DataFrame(
            {
                "orthogroup": ["og1", "og1", "og1", "og1"],
                "branch_id": [0, 0, 1, 2],
                "parent": [1, 2, 2, 2],
                "flag": [1, 0, 0, 0],
                "value": [11, 99, 20, 30],
            }
        )
        self.assertEqual(kfog.get_most_recent(b_dup, 0, "og1", "flag", 1, "value"), 11)

        d = pandas.DataFrame(
            {
                "branch_id": [0, 1, 2],
                "parent": [1, 2, 2],
                "x": [10.0, 13.0, 20.0],
            }
        )
        out_d = kfog.compute_delta(d, "x")
        self.assertAlmostEqual(out_d.loc[0, "delta_x"], -3.0)
        self.assertAlmostEqual(out_d.loc[1, "delta_x"], -7.0)
        self.assertAlmostEqual(out_d.loc[2, "delta_x"], 0.0)

    def test_kfplot(self):
        df = pandas.DataFrame(
            {"v": numpy.random.normal(size=80), "g": ["A"] * 40 + ["B"] * 40}
        )
        ax = kfplot.hist_boxplot(x="v", category="g", df=df, xlim=[-3, 3])
        self.assertIsNotNone(ax)
        import matplotlib.pyplot as plt
        df_bar = pandas.DataFrame({"x1": [1, 2], "x2": [2, 3], "y": ["A", "B"]})
        fig2, ax2 = plt.subplots()
        out_ax2 = kfplot.stacked_barplot(x=["x1", "x2"], y="y", data=df_bar, colors=["C0", "C1"], ax=ax2)
        self.assertIsNotNone(out_ax2)
        colors = ["C0", "C1"]
        fig3, ax3 = plt.subplots()
        _ = kfplot.hist_boxplot(x="v", category="g", df=df, colors=colors, xlim=[-3, 3], ax=ax3)
        self.assertEqual(colors, ["C0", "C1"])
        plt.close(fig3)
        fig4, ax4 = plt.subplots()
        out_ax4 = kfplot.density_scatter(
            x=df["v"].values,
            y=(df["v"].values * 0.5) + numpy.random.normal(scale=0.1, size=df.shape[0]),
            ax=ax4,
            cbar=False,
            show_cor_p=True,
        )
        self.assertIsNotNone(out_ax4)
        plt.close(fig4)
        plt.close(fig2)
        plt.close(ax.figure)

    def test_kfplot_density_scatter_empty(self):
        with self.assertRaises(ValueError):
            kfplot.density_scatter(x=[numpy.nan], y=[numpy.nan], cbar=False)

    def test_kfstat(self):
        x = numpy.random.normal(size=200)
        y = numpy.random.normal(loc=0.2, size=200)
        out = kfstat.bm_test(x, y)
        self.assertEqual(len(out), 6)


if __name__ == "__main__":
    unittest.main()
