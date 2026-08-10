import os
import re
import tempfile
import unittest

import ete4
import matplotlib
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kftools import kfexpression, kfog, kfphylo, kfplot, kfseq, kfspecies


class TestKFToolsRegressions(unittest.TestCase):
    def test_tau_uses_standard_n_minus_one_denominator(self):
        data = pd.DataFrame({"expressed": [1.0], "silent": [0.0]})
        tau = kfexpression.calc_tau(
            data,
            ["expressed", "silent"],
            unlog2=False,
            unPlus1=False,
        )
        self.assertEqual(tau.tolist(), [1.0])

        one_tissue = kfexpression.calc_tau(
            data,
            ["expressed"],
            unlog2=False,
            unPlus1=False,
        )
        self.assertEqual(one_tissue.tolist(), [0.0])

        with self.assertRaisesRegex(ValueError, "non-negative"):
            kfexpression.calc_tau(
                pd.DataFrame({"a": [-1.0], "b": [-2.0]}),
                ["a", "b"],
                unlog2=False,
                unPlus1=False,
            )

    def test_complementarity_is_symmetric_and_validated(self):
        forward = kfexpression.calc_complementarity([1.0, 0.0], [0.0, 1.0])
        reverse = kfexpression.calc_complementarity([0.0, 1.0], [1.0, 0.0])
        self.assertEqual(forward, reverse)
        with self.assertRaisesRegex(ValueError, "same number of values"):
            kfexpression.calc_complementarity([1.0], [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            kfexpression.calc_complementarity([-1.0], [-2.0])

    def test_f3x4_rejects_incomplete_codons(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write(">A\nATGC\n")
        try:
            with self.assertRaisesRegex(ValueError, "multiple of three"):
                kfseq.alignment2nuc_freqs("A", path, "F3X4")
        finally:
            os.unlink(path)

    def test_theta_values_are_normalized_and_bounded(self):
        theta = kfseq.nuc_freq2theta([{"A": 1.0, "T": 1.0, "C": 1.0, "G": 1.0}])
        self.assertEqual(theta, [{"theta": 0.5, "theta1": 0.5, "theta2": 0.5}])
        with self.assertRaisesRegex(ValueError, "positive total"):
            kfseq.nuc_freq2theta([{"A": 0.0, "T": 0.0, "C": 0.0, "G": 0.0}])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfseq.get_mapnh_thetas(
                "F3X4",
                [{"theta": 2.0, "theta1": 0.5, "theta2": 0.5}],
            )

    def test_polytomy_weighting_uses_every_child(self):
        tree = ete4.PhyloTree("(A:1,B:2,C:3);", parser=1)

        def values(middle):
            return {
                "A": [{"theta": 0.1}] * 3,
                "B": [{"theta": middle}] * 3,
                "C": [{"theta": 0.9}] * 3,
            }

        low_middle = kfseq.weighted_mean_root_thetas(values(0.2), tree, "F3X4")[0]["theta"]
        high_middle = kfseq.weighted_mean_root_thetas(values(0.8), tree, "F3X4")[0]["theta"]
        self.assertNotEqual(low_middle, high_middle)

    def test_internal_node_names_are_unique(self):
        tree = ete4.PhyloTree("((A:1,B:1)n1:1,(C:1,D:1):1);", parser=1)
        filled = kfphylo.fill_internal_node_names(tree)
        internal_names = [node.name for node in filled.traverse() if not node.is_leaf]
        self.assertEqual(len(internal_names), len(set(internal_names)))

    def test_ultrametric_zero_tolerance_is_exact(self):
        self.assertFalse(kfphylo.check_ultrametric("(A:1000,B:999.5);"))

    def test_transfer_root_does_not_mutate_input_on_failure(self):
        source = ete4.PhyloTree(
            "((A:1,B:1):-1,(C:1,D:1):1);",
            parser=1,
        )
        target = ete4.PhyloTree(
            "(A:1,(B:1,(C:1,D:1):1):1);",
            parser=1,
        )
        target.dist = 7.0
        split_before = [set(child.leaf_names()) for child in target.children]
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            kfphylo.transfer_root(target, source)
        self.assertEqual(target.dist, 7.0)
        self.assertEqual(
            [set(child.leaf_names()) for child in target.children],
            split_before,
        )

    def test_nwk2table_preserves_mixed_numeric_attributes(self):
        tree = ete4.PhyloTree("(A:1,B:1);", parser=1)
        tree.foo = 2.75
        tree.children[0].foo = 1
        tree.children[1].foo = 1.5
        table = kfog.nwk2table(tree, attr="foo")
        self.assertEqual(sorted(table["foo"].tolist()), [1.0, 1.5, 2.75])

    def test_density_diagonal_uses_only_endpoints(self):
        ax = kfplot.density_scatter(
            [0.0, 1e12],
            [0.0, 1e12],
            diag=True,
            cbar=False,
        )
        try:
            diagonal = ax.lines[-1]
            self.assertEqual(len(diagonal.get_xdata()), 2)
            self.assertEqual(len(diagonal.get_ydata()), 2)
        finally:
            plt.close(ax.figure)

    def test_regressions_accept_non_identifier_column_names(self):
        data = pd.DataFrame({"x value": [1.0, 2.0, 3.0, 4.0], "y value": [2.0, 3.5, 6.0, 7.0]})
        density_ax = kfplot.density_scatter(
            "x value",
            "y value",
            df=data,
            reg_family=sm.families.Gaussian(),
            cbar=False,
        )
        annotation_ax = kfplot.ols_annotations(
            "x value",
            "y value",
            data=data,
            method="quantreg",
        )
        plt.close(density_ax.figure)
        plt.close(annotation_ax.figure)

    def test_ragged_alignment_is_rejected(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            path = tmp.name
            tmp.write(">a\nAAAA\n>b\nAA\n")
        try:
            with self.assertRaisesRegex(ValueError, "same aligned length"):
                kfog.get_aln_stats(path)
        finally:
            os.unlink(path)

    def test_regex_dict_preserves_group_zero(self):
        label = "prefix_Homo_sapiens_suffix"
        mapping_config = {
            "type": "regex",
            "pattern": r"prefix_(Homo_sapiens)",
            "group": 0,
        }
        tuple_config = ("regex", r"prefix_(Homo_sapiens)", 0)
        self.assertEqual(
            kfspecies.parse_species_label(label, species_parser=mapping_config),
            kfspecies.parse_species_label(label, species_parser=tuple_config),
        )

    def test_species_result_properties_and_derived_names(self):
        plain = kfspecies.SpeciesParseResult("Homo_sapiens")
        proximity = kfspecies.SpeciesParseResult("Homo_cf_sapiens")
        unknown = kfspecies.SpeciesParseResult("Amoeba_sp_isolate7")
        ranked = kfspecies.SpeciesParseResult("Bacillus_subtilis_var_168")
        genus_only = kfspecies.SpeciesParseResult("Amoeba")

        self.assertEqual((plain.genus, plain.species), ("Homo", "sapiens"))
        self.assertEqual(proximity.scientific_name, "Homo cf. sapiens")
        self.assertEqual(proximity.taxonomy_query, "Homo sapiens")
        self.assertEqual(unknown.species, "isolate7")
        self.assertEqual(unknown.taxonomy_query, "Amoeba")
        self.assertEqual(ranked.scientific_name, "Bacillus subtilis var. 168")
        self.assertEqual(ranked.taxonomy_query, "Bacillus subtilis")
        self.assertEqual(genus_only.species, "")
        with self.assertRaisesRegex(ValueError, "species_label"):
            kfspecies.SpeciesParseResult("  ")

    def test_taxonomic_parser_variants_and_failures(self):
        cases = {
            "homo_sapiens": "Homo_sapiens",
            "homo_cf_sapiens": "Homo_cf_sapiens",
            "homo_sapiens_aff": "Homo_aff_sapiens",
            "amoeba_spp.": "Amoeba_sp",
            "bacillus_subtilis_ssp_168": "Bacillus_subtilis_subsp_168",
            "prefix|Mus_musculus|suffix": "Mus_musculus",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                parsed = kfspecies.parse_species_label(
                    label,
                    species_parser="taxonomic",
                )
                self.assertEqual(parsed.species_label, expected)

        for label in [None, "", "single", "Homo_cf", "Homo_sapiens_var"]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                kfspecies.parse_species_label(label, species_parser="taxonomic")

    def test_callable_parser_result_coercion(self):
        results = [
            kfspecies.SpeciesParseResult("Homo_sapiens"),
            {"genus": "Homo", "species": "sapiens"},
            {"species_label": "Homo_sapiens"},
            ("Homo", "sapiens"),
            "Homo_sapiens",
        ]
        for result in results:
            with self.subTest(result=result):
                parsed = kfspecies.parse_species_label(
                    "ignored_label",
                    species_parser=lambda _label, value=result: value,
                )
                self.assertEqual(parsed.species_label, "Homo_sapiens")

        invalid_results = [{}, ["Homo"], 1]
        for result in invalid_results:
            with self.subTest(result=result), self.assertRaises(ValueError):
                kfspecies.parse_species_label(
                    "ignored_label",
                    species_parser=lambda _label, value=result: value,
                )

    def test_regex_and_map_parser_configurations(self):
        label = "id=Homo_sapiens"
        regex_configs = [
            re.compile(r"(?P<genus>Homo)_(?P<species>sapiens)"),
            r"Homo_sapiens",
            r"(Homo)_(sapiens)",
            r"id=(Homo_sapiens)",
            ("regex", r"(Homo)_(sapiens)", (1, 2)),
            {"pattern": r"(Homo)_(sapiens)", "groups": (1, 2)},
        ]
        for config in regex_configs:
            with self.subTest(config=config):
                parsed = kfspecies.parse_species_label(label, species_parser=config)
                self.assertEqual(parsed.species_label, "Homo_sapiens")

        mapping = {label: {"label": "Homo_sapiens"}}
        for config in [
            {"type": "map", "mapping": mapping},
            ("map", mapping),
            ["legacy"],
            {"mode": "taxonomic"},
        ]:
            parsed = kfspecies.parse_species_label(label, species_parser=config)
            if config == ["legacy"]:
                self.assertEqual(parsed.species_label, "id=Homo_sapiens")
            else:
                self.assertEqual(parsed.species_label, "Homo_sapiens")

        invalid_configs = [
            {"type": "regex"},
            {"type": "map", "mapping": None},
            {"type": "unknown"},
            [],
            ["regex"],
            ["map"],
            ("regex", r"(Homo)_(sapiens)", (1,)),
            1,
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                kfspecies.parse_species_label(label, species_parser=config)

        with self.assertRaisesRegex(ValueError, "did not match"):
            kfspecies.parse_species_label("Mus_musculus", species_parser=r"Homo_sapiens")
        with self.assertRaisesRegex(ValueError, "did not match map"):
            kfspecies.parse_species_label("Mus_musculus", species_parser=("map", mapping))
        with self.assertRaisesRegex(ValueError, "Use only one"):
            kfspecies.parse_species_label(
                label,
                species_parser="legacy",
                parser="taxonomic",
            )

    def test_get_most_recent_handles_missing_target_values(self):
        data = pd.DataFrame(
            {
                "orthogroup": ["og", "og"],
                "branch_id": [0, 1],
                "parent": [1, pd.NA],
                "flag": [pd.NA, 1],
                "value": [10, 20],
            }
        )
        self.assertEqual(
            kfog.get_most_recent(data, 0, "og", "flag", 1, "value"),
            20,
        )
        self.assertEqual(
            kfog.get_most_recent(data, 0, "og", "flag", pd.NA, "value"),
            10,
        )
