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


@pytest.mark.parametrize(
    ("target", "source", "expected_names"),
    [
        ("(A:1,B:1,C:1)target;", "(C:1,A:1,B:1)source;", {"source"}),
        (
            "((A:1,B:1,C:1)target_abc:1,D:2)target_root;",
            "((C:1,A:1,B:1)source_abc:1,D:2)source_root;",
            {"source_abc", "source_root"},
        ),
    ],
)
def test_transfer_internal_node_names_supports_polytomies(target, source, expected_names):
    result = kfphylo.transfer_internal_node_names(target, source)
    assert {node.name for node in result.traverse() if not node.is_leaf} == expected_names


def test_transfer_internal_node_names_preserves_unary_node_multiplicity():
    result = kfphylo.transfer_internal_node_names(
        "(((A:1)target_inner:1)target_outer:1,B:3)target_root;",
        "(((A:1)source_inner:1)source_outer:1,B:3)source_root;",
    )
    assert [node.name for node in result.traverse() if not node.is_leaf] == [
        "source_root",
        "source_outer",
        "source_inner",
    ]

    with pytest.raises(ValueError, match="clade_multiplicity_mismatches"):
        kfphylo.transfer_internal_node_names(
            "(((A:1)target_inner:1)target_outer:1,B:3)target_root;",
            "((A:2)source_single:1,B:3)source_root;",
        )


def test_transfer_root_preserves_multifurcating_partition_and_inputs():
    target = ete4.PhyloTree("((A:1,B:1)AB:2,(C:1,D:1)CD:2)target_root;", parser=1)
    source = ete4.PhyloTree("((A:1,B:1)source_ab:2,C:1,D:1)source_root;", parser=1)
    target_before = target.write(parser=1, format_root_node=True)
    source_before = source.write(parser=1, format_root_node=True)

    result = kfphylo.transfer_root(target, source)

    assert len(result.children) == 3
    assert {frozenset(child.leaf_names()) for child in result.children} == {
        frozenset({"A", "B"}),
        frozenset({"C"}),
        frozenset({"D"}),
    }
    distance_by_clade = {frozenset(child.leaf_names()): child.dist for child in result.children}
    assert distance_by_clade[frozenset({"A", "B"})] == pytest.approx(4.0)
    assert distance_by_clade[frozenset({"C"})] == pytest.approx(1.0)
    assert distance_by_clade[frozenset({"D"})] == pytest.approx(1.0)
    assert target.write(parser=1, format_root_node=True) == target_before
    assert source.write(parser=1, format_root_node=True) == source_before


def test_transfer_root_supports_four_way_root_and_rejects_missing_vertex():
    source = "(A:1,B:2,C:3,D:4)source_root;"
    result = kfphylo.transfer_root("(D:4,C:4,B:4,A:4)target_root;", source)
    assert len(result.children) == 4
    assert {frozenset(child.leaf_names()) for child in result.children} == {frozenset({name}) for name in "ABCD"}

    with pytest.raises(ValueError, match="does not contain a unique vertex"):
        kfphylo.transfer_root("((A:1,B:1):1,(C:1,D:1):1);", source)


def test_transfer_root_preserves_pairwise_distances_at_multifurcating_vertex():
    target = ete4.PhyloTree("(A:10,B:1,C:1)target_root;", parser=1)
    source = ete4.PhyloTree("(A:1,B:1,C:1)source_root;", parser=1)
    leaves_before = {leaf.name: leaf for leaf in target.leaves()}
    distances_before = {
        pair: target.get_distance(leaves_before[pair[0]], leaves_before[pair[1]])
        for pair in [("A", "B"), ("A", "C"), ("B", "C")]
    }

    result = kfphylo.transfer_root(target, source)

    leaves_after = {leaf.name: leaf for leaf in result.leaves()}
    distances_after = {
        pair: result.get_distance(leaves_after[pair[0]], leaves_after[pair[1]]) for pair in distances_before
    }
    assert distances_after == pytest.approx(distances_before)
    assert {child.name: child.dist for child in result.children} == {"A": 10.0, "B": 1.0, "C": 1.0}


def test_transfer_root_validates_multifurcating_source_distances_without_applying_them():
    source = ete4.PhyloTree("(A:1,B:1,C:1)source_root;", parser=1)
    source.children[0].dist = -1.0
    with pytest.raises(ValueError, match="tree_from root child branch lengths must be non-negative"):
        kfphylo.transfer_root("(A:10,B:1,C:1)target_root;", source)


def test_multifurcating_age_outputs_are_invariant_to_child_order():
    first_tree = "((A_a:1,B_b:1,C_c:1)ABC:1,D_d:2)Root;"
    reordered_tree = "(D_d:2,(C_c:1,A_a:1,B_b:1)ABC:1)Root;"
    first_table = kfog.nwk2table(first_tree, attr="dist", age=True).set_index("branch_id")
    reordered_table = kfog.nwk2table(reordered_tree, attr="dist", age=True).set_index("branch_id")
    pd.testing.assert_series_equal(first_table["age"].sort_index(), reordered_table["age"].sort_index())

    first_mapping = kfog.node_gene2species(
        "((A_a_g1:1,B_b_g1:1,C_c_g1:1)GABC:1,D_d_g1:2)GRoot;",
        first_tree,
        is_ultrametric=True,
    )
    reordered_mapping = kfog.node_gene2species(
        "(D_d_g1:2,(C_c_g1:1,A_a_g1:1,B_b_g1:1)GABC:1)GRoot;",
        reordered_tree,
        is_ultrametric=True,
    )
    pd.testing.assert_frame_equal(
        first_mapping.sort_values("branch_id").reset_index(drop=True),
        reordered_mapping.sort_values("branch_id").reset_index(drop=True),
    )


def test_ou2table_preserves_all_multifurcating_sister_comparisons(tmp_path):
    regime_path = tmp_path / "regimes.tsv"
    regime_path.write_text("node_name\tregime\nA_x\t1\nB_x\t2\n")
    leaf_path = tmp_path / "leaf.tsv"
    leaf_path.write_text(
        "node_name\tparam\tregime\tt1\tt2\nx\tmu\t0\t1.0\t2.0\nx\tmu\t1\t3.0\t4.0\nx\tmu\t2\t5.0\t6.0\n"
    )
    first_tree_path = tmp_path / "first.nwk"
    first_tree_path.write_text("(A_x:1,B_x:1,C_x:1)Root;\n")
    reordered_tree_path = tmp_path / "reordered.nwk"
    reordered_tree_path.write_text("(C_x:1,A_x:1,B_x:1)Root;\n")

    first = kfog.ou2table(regime_path, leaf_path, first_tree_path).sort_values("branch_id").reset_index(drop=True)
    reordered = (
        kfog.ou2table(regime_path, leaf_path, reordered_tree_path).sort_values("branch_id").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(first, reordered)

    names = kfog.nwk2table(first_tree_path, attr="name")
    label_by_name = dict(zip(names["name"], names["branch_id"], strict=True))
    a_row = first.loc[first["branch_id"] == label_by_name["A_x"]].iloc[0]
    expected_sisters = tuple(sorted([label_by_name["B_x"], label_by_name["C_x"]]))
    assert a_row["sister_branch_ids"] == expected_sisters
    assert dict(zip(a_row["sister_branch_ids"], a_row["delta_maxmu_sisters"], strict=True)) == {
        label_by_name["B_x"]: -2.0,
        label_by_name["C_x"]: 2.0,
    }
    assert len(a_row["mu_complementarity_sisters"]) == 2
    assert pd.isna(a_row["delta_maxmu"])
    assert pd.isna(a_row["mu_complementarity"])
    assert a_row["delta_maxmu_parent"] == 2.0
    assert not pd.isna(a_row["mu_complementarity_parent"])


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

    reordered_path = tmp_path / "polytomy-reordered.nwk"
    reordered_path.write_text("(C_c_1:1,A_a_1:1,B_b_1:1)Root;")
    reordered_relations = kfog.nwk2table(reordered_path, attr="name", sister=True)
    pd.testing.assert_frame_equal(
        relation_table.sort_values("branch_id").reset_index(drop=True),
        reordered_relations.sort_values("branch_id").reset_index(drop=True),
    )
    reordered_statistics = kfog.get_misc_node_statistics(reordered_path)
    relationship_columns = ["branch_id", "parent", "sister", "sisters", "child1", "child2", "children"]
    pd.testing.assert_frame_equal(
        statistics.loc[:, relationship_columns].sort_values("branch_id").reset_index(drop=True),
        reordered_statistics.loc[:, relationship_columns].sort_values("branch_id").reset_index(drop=True),
    )


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
