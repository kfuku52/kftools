import unittest

import matplotlib

matplotlib.use("Agg")

from kftools import kfspecies


class TestKFSpecies(unittest.TestCase):
    def test_taxonomic_punctuation_and_scientific_name_round_trips(self):
        labels = [
            "Quercus_cf_robur",
            "Quercus_aff_robur",
            "Quercus_nr_robur",
            "Bacillus_subtilis_subsp_168",
            "Brassica_oleracea_var_capitata",
            "Acer_palmatum_forma_atropurpureum",
            "Amoeba_sp_JD1",
            "Bacillus_subtilis_strain_168.1",
        ]
        for label in labels:
            with self.subTest(label=label):
                parsed = kfspecies.parse_species_label(label, species_parser="taxonomic")
                reparsed = kfspecies.parse_species_label(parsed.scientific_name, species_parser="taxonomic")
                self.assertEqual(parsed, reparsed)
        for marker in ["ssp.", "subspecies.", "SUBSP."]:
            with self.subTest(marker=marker):
                result = kfspecies.parse_species_label(f"Bacillus subtilis {marker} 168", species_parser="taxonomic")
                self.assertEqual(result.species_label, "Bacillus_subtilis_subsp_168")

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
