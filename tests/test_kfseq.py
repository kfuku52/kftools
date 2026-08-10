import os
import tempfile
import unittest

import ete4
import matplotlib
import numpy as np

matplotlib.use("Agg")

from kftools import kfseq


class TestKFSeq(unittest.TestCase):
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
        expected_theta3 = ((0.1 / 1.0) + (0.5 / 2.0) + (0.9 / 3.0)) / ((1.0 / 1.0) + (1.0 / 2.0) + (1.0 / 3.0))
        self.assertAlmostEqual(root_thetas3[0]["theta"], expected_theta3)
        self.assertEqual(kfseq.get_mapnh_thetas("F3X4", []), "F3X4()")

    def test_kfseq_value_and_path_validation(self):
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

    def test_alignment2nuc_freqs_fasta_validation(self):
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

    def test_weighted_mean_root_thetas_validation(self):
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
        self.assertAlmostEqual(root_thetas3[0]["theta"], (0.1 + 0.3 + 0.9) / 3)

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
