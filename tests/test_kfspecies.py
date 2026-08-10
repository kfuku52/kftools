import unittest

import ete4
import matplotlib

matplotlib.use("Agg")

from kftools import kfphylo, kfseq, kfspecies


class TestKFSpecies(unittest.TestCase):
    def test_kfspecies_taxonomic_parser_supports_natural_order_labels(self):
        proximity = kfspecies.parse_species_label(
            "Dictyostelium_discoideum_cf_gene1",
            species_parser="taxonomic",
        )
        self.assertEqual(proximity.species_label, "Dictyostelium_cf_discoideum")
        self.assertEqual(proximity.scientific_name, "Dictyostelium cf. discoideum")
        self.assertEqual(proximity.taxonomy_query, "Dictyostelium discoideum")

        genus_only = kfspecies.parse_species_label(
            "Amoeba_sp_JDSRuffled_gene2",
            species_parser="taxonomic",
        )
        self.assertEqual(genus_only.species_label, "Amoeba_sp_JDSRuffled")
        self.assertEqual(genus_only.taxonomy_query, "Amoeba")

        ranked = kfspecies.parse_species_label(
            "Bacillus_subtilis_subsp_168_gene3",
            species_parser="taxonomic",
        )
        self.assertEqual(ranked.species_label, "Bacillus_subtilis_subsp_168")
        self.assertEqual(ranked.scientific_name, "Bacillus subtilis subsp. 168")
        self.assertEqual(ranked.taxonomy_query, "Bacillus subtilis")

    def test_species_parser_changes_do_not_regress_safe_utilities(self):
        tree_from = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        tree_to = ete4.PhyloTree("((A:1,B:1):2,(C:1,D:1):2);", parser=1)
        rerooted = kfphylo.transfer_root(tree_to=tree_to, tree_from=tree_from)
        self.assertEqual(
            {frozenset(child.leaf_names()) for child in rerooted.children},
            {frozenset(["A", "B"]), frozenset(["C", "D"])},
        )

        nuc_freqs = [
            {"A": 0.25, "T": 0.25, "C": 0.25, "G": 0.25},
            {"A": 0.10, "T": 0.40, "C": 0.20, "G": 0.30},
            {"A": 0.30, "T": 0.20, "C": 0.10, "G": 0.40},
        ]
        thetas = kfseq.nuc_freq2theta(nuc_freqs)
        self.assertEqual(len(thetas), 3)
        self.assertTrue(all("theta" in theta_row for theta_row in thetas))

        tree = ete4.PhyloTree("(A:2,B:1);", parser=1)
        subroot_thetas = {
            "A": [{"theta": 0.1, "theta1": 0.3, "theta2": 0.7}] * 3,
            "B": [{"theta": 0.9, "theta1": 0.7, "theta2": 0.3}] * 3,
        }
        root_thetas = kfseq.weighted_mean_root_thetas(subroot_thetas, tree, model="F3X4")
        self.assertAlmostEqual(root_thetas[0]["theta"], 0.1 + (0.9 - 0.1) * (2.0 / 3.0))
