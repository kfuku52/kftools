import gzip
import os
import tempfile
import unittest
from pathlib import Path

import ete4
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from kftools import kfog, kfphylo


class TestKFOG(unittest.TestCase):
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

    def test_kfog_node_gene2species_species_parsers(self):
        def coverage_signature(df):
            signature = []
            for value in df.sort_values("branch_id")["spnode_coverage"].tolist():
                tokens = [token for token in str(value).split(",") if token != ""]
                signature.append(len(tokens))
            return signature

        species_tree_legacy = ete4.PhyloTree("((A_x:1,B_x:1):1,C_x:2);", parser=1)
        gene_tree_legacy = ete4.PhyloTree("((A_x_g1:1,B_x_g2:1):1,C_x_g3:2);", parser=1)
        out_legacy = kfog.node_gene2species(
            gene_tree_legacy,
            species_tree_legacy,
            is_ultrametric=False,
        )

        species_tree_taxonomic = ete4.PhyloTree(
            "((Dictyostelium_cf_discoideum:1,Amoeba_sp_JDSRuffled:1):1,Bacillus_subtilis_subsp_168:2);",
            parser=1,
        )
        gene_tree_taxonomic = ete4.PhyloTree(
            "((Dictyostelium_cf_discoideum_g1:1,Amoeba_sp_JDSRuffled_g2:1):1,Bacillus_subtilis_subsp_168_g3:2);",
            parser=1,
        )
        out_taxonomic = kfog.node_gene2species(
            gene_tree_taxonomic,
            species_tree_taxonomic,
            is_ultrametric=False,
            species_parser="taxonomic",
        )

        regex_parser = {
            "type": "regex",
            "pattern": r"sp(?P<genus>[A-Z])_(?P<species>[a-z])zz(?:_g\d+)?",
        }
        species_tree_regex = ete4.PhyloTree("((spA_xzz:1,spB_xzz:1):1,spC_xzz:2);", parser=1)
        gene_tree_regex = ete4.PhyloTree("((spA_xzz_g1:1,spB_xzz_g2:1):1,spC_xzz_g3:2);", parser=1)
        out_regex = kfog.node_gene2species(
            gene_tree_regex,
            species_tree_regex,
            is_ultrametric=False,
            species_parser=regex_parser,
        )

        legacy_signature = coverage_signature(out_legacy)
        self.assertEqual(sorted(legacy_signature), sorted(coverage_signature(out_taxonomic)))
        self.assertEqual(sorted(legacy_signature), sorted(coverage_signature(out_regex)))
        self.assertEqual(
            {value for value in out_taxonomic["spnode_coverage"].tolist() if value != ""},
            {
                "Amoeba_sp_JDSRuffled",
                "Bacillus_subtilis_subsp_168",
                "Dictyostelium_cf_discoideum",
            },
        )

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

    def test_kfog_compute_delta_input_validation(self):
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

    def test_kfog_regime2tree_parameter_validation(self):
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

    def test_kfog_ou2table_mapping_validation(self):
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

    def test_kfog_ou2table_regime_id_validation(self):
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
