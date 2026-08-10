import re

import ete4
import pandas as pd
import pytest

from kftools import kfog, kfphylo, kfseq, kfspecies


def test_invalid_species_regex_has_stable_value_error():
    with pytest.raises(ValueError, match="invalid regex species parser pattern"):
        kfspecies.parse_species_label("Homo_sapiens", species_parser={"type": "regex", "pattern": "("})


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("scientific_name", ""), ("taxonomy_query", 42)],
)
def test_species_parse_result_rejects_invalid_explicit_names(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        kfspecies.SpeciesParseResult("Homo_sapiens", **{keyword: value})


@pytest.mark.parametrize(
    ("model", "count", "expected"),
    [("F1X4", 3, 1), ("F3X4", 1, 3), ("F3X4", 2, 3)],
)
def test_mapnh_theta_cardinality_matches_frequency_model(model, count, expected):
    theta = {"theta": 0.5, "theta1": 0.5, "theta2": 0.5}
    with pytest.raises(ValueError, match=rf"requires either 0 or {expected} theta entries"):
        kfseq.get_mapnh_thetas(model, [theta] * count)


def test_transfer_internal_node_names_is_non_destructive_on_success():
    target = ete4.PhyloTree("((A:1,B:1)target_ab:1,(C:1,D:1)target_cd:1)target_root;", parser=1)
    source = ete4.PhyloTree("((A:1,B:1)source_ab:1,(C:1,D:1)source_cd:1)source_root;", parser=1)
    target_names = [node.name for node in target.traverse()]
    source_names = [node.name for node in source.traverse()]

    result = kfphylo.transfer_internal_node_names(target, source)

    assert [node.name for node in target.traverse()] == target_names
    assert [node.name for node in source.traverse()] == source_names
    assert {node.name for node in result.traverse() if not node.is_leaf} == {
        "source_ab",
        "source_cd",
        "source_root",
    }


def test_polytomy_relationship_columns_are_lossless(tmp_path):
    tree = "(A_a_1:1,B_b_1:1,C_c_1:1)Root;"
    relation_table = kfog.nwk2table(tree, attr="name", sister=True)
    leaf_row = relation_table.loc[relation_table["name"] == "A_a_1"].iloc[0]
    assert len(leaf_row["sisters"]) == 2
    assert leaf_row["sister"] in leaf_row["sisters"]

    tree_path = tmp_path / "polytomy.nwk"
    tree_path.write_text(tree)
    statistics = kfog.get_misc_node_statistics(tree_path)
    root_row = statistics.loc[statistics["parent"] == -999].iloc[0]
    leaf_rows = statistics.loc[statistics["num_leaf"] == 1]
    assert len(root_row["children"]) == 3
    assert all(len(sisters) == 2 for sisters in leaf_rows["sisters"])


def test_prepared_most_recent_lookup_matches_scalar_api_and_reuses_indexes():
    data = pd.DataFrame(
        {
            "orthogroup": ["og1", "og1", "og1", "og2", "og2"],
            "branch_id": [0, 1, 2, 0, 1],
            "parent": [1, 2, 2, 1, 1],
            "flag": [0, 0, 1, 0, 1],
            "value": [10, 20, 30, 40, 50],
        }
    )
    lookup = kfog.prepare_most_recent_lookup(data, "flag", "value")
    prepared_table_ids = {orthogroup: id(table) for orthogroup, table in lookup.tables.items()}

    for branch_id, orthogroup in [(0, "og1"), (1, "og1"), (0, "og2")]:
        assert lookup.find(branch_id, orthogroup, 1) == kfog.get_most_recent(
            data,
            branch_id,
            orthogroup,
            "flag",
            1,
            "value",
        )
    assert {orthogroup: id(table) for orthogroup, table in lookup.tables.items()} == prepared_table_ids


def test_missing_gene_species_uses_warning_instead_of_stderr():
    with pytest.warns(RuntimeWarning, match=re.escape("['D_x']")):
        result = kfog.node_gene2species(
            "((A_x_g1:1,B_x_g2:1):1,D_x_g3:2);",
            "((A_x:1,B_x:1):1,C_x:2);",
        )
    assert result.shape[0] == 5
