"""Column names, text identifiers, and dataframe preservation contracts."""

import numpy as np
import pandas as pd
import pytest

from kftools import kfog


def branch_table():
    return pd.DataFrame(
        {
            "orthogroup": ["og", "og", "og"],
            "branch_id": [0, 1, 2],
            "parent": [1, 2, -1],
            "flag": [0, 1, 1],
            "value": [10.0, 20.0, 30.0],
        },
        index=[9, 3, 7],
    )


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    ("return_col", "expected"),
    [
        ("branch_id", 1),
        ("parent", 2),
        ("flag", 1),
        ("orthogroup", "og"),
        ("value", 20),
    ],
)
def test_ancestor_lookup_accepts_overlapping_column_roles(prepared, return_col, expected):
    data = branch_table()
    before = data.copy(deep=True)
    if prepared:
        lookup = kfog.prepare_most_recent_lookup(data, "flag", return_col)
        result = lookup.find(0, "og", 1)
    else:
        result = kfog.get_most_recent(data, 0, "og", "flag", 1, return_col)
    assert result == expected
    pd.testing.assert_frame_equal(data, before)


def test_ancestor_lookup_can_use_structural_columns_as_targets():
    data = branch_table()
    for target in ["parent", "branch_id"]:
        scalar = kfog.get_most_recent(data, 0, "og", target, 2, target)
        lookup = kfog.prepare_most_recent_lookup(data, target, target)
        assert scalar == lookup.find(0, "og", 2) == 2
    # The orthogroup column itself can be both target and result.
    lookup = kfog.prepare_most_recent_lookup(data, "orthogroup", "orthogroup")
    assert lookup.find(0, "og", "og") == "og"


def test_compute_delta_preserves_existing_columns_index_and_input():
    data = branch_table().assign(parent_value=[99.0, 88.0, 77.0])
    before = data.copy(deep=True)
    result = kfog.compute_delta(data, "value")
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(result[before.columns], before)
    np.testing.assert_allclose(result["delta_value"], [-10.0, -10.0, np.nan])


@pytest.mark.parametrize("node_name", ["4", "004", "NA", "NaN", "null"])
def test_ou_reads_node_names_as_lossless_identifiers(tmp_path, node_name):
    tree = tmp_path / "tree.nwk"
    regimes = tmp_path / "regimes.tsv"
    traits = tmp_path / "traits.tsv"
    tree.write_text(f"((A:1,B:1){node_name}:1,C:2)Root;")
    regimes.write_text(f"node_name\tregime\n{node_name}\t1\n")
    traits.write_text("node_name\tparam\tregime\tleaf\troot\nC\tmu\t0\t1\t1\nA\tmu\t1\t2\t1\nB\tmu\t1\t2\t1\n")
    result = kfog.ou2table(regimes, traits, tree)
    assert len(result) == 5
    assert result["is_shift"].sum() == 1
