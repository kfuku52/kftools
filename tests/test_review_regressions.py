"""Numerical precision, identifiers, and input ambiguity regressions."""

from types import MappingProxyType

import matplotlib
import numpy as np
import pandas as pd
import pytest
from ete4 import PhyloTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from kftools import kfexpression, kfog, kfphylo, kfplot, kfseq


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("UInt8", [10, 1], -9),
        ("uint8", [10, 1], -9),
        ("UInt64", [2**64 - 1, 0], -(2**64 - 1)),
        ("Int64", [-(2**63), 2**63 - 1], 2**64 - 1),
        ("int64", [2**53, 2**53 + 1], 1),
        ("Float64", [10, 1], -9),
    ],
)
def test_delta_preserves_exact_integer_differences(dtype, values, expected):
    data = pd.DataFrame({"branch_id": [0, 1], "parent": [-1, 0], "v": pd.Series(values, dtype=dtype)})
    before = data.copy(deep=True)
    result = kfog.compute_delta(data, "v")
    assert pd.isna(result.loc[0, "delta_v"])
    assert result.loc[1, "delta_v"] == expected
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_series_equal(result.v, data.v)


@pytest.mark.parametrize("dtype", ["UInt8", "Int64", "Float64"])
def test_delta_nullable_missing_values(dtype):
    data = pd.DataFrame({"branch_id": [0, 1, 2], "parent": [-1, 0, 1], "v": pd.Series([10, None, 1], dtype=dtype)})
    assert kfog.compute_delta(data, "v").delta_v.isna().all()


def test_tau_rejects_ambiguous_selected_columns():
    data = pd.DataFrame([[10, 10, 0]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column"):
        kfexpression.calc_tau(data, ["a", "b"], unlog2=False)
    np.testing.assert_array_equal(kfexpression.calc_tau(data, ["b"], unlog2=False), [0])


@pytest.mark.parametrize("internal_name", ["", "same"])
def test_species_mapping_ids_distinguish_internal_nodes(internal_name):
    species = PhyloTree(f"((A_x:1,B_x:1){internal_name}:1,C_x:2){internal_name};", parser=1)
    gene = PhyloTree("((A_x_g:1,B_x_g:1):1,C_x_g:2);", parser=1)
    species_before, gene_before = species.write(), gene.write()
    result = kfog.node_gene2species(gene, species, is_ultrametric=True).set_index("branch_id")
    reference = kfog.nwk2table(species.write(), attr="name").set_index("branch_id")
    assert set(result.spnode_coverage_id) == set(reference.index)
    assert result.spnode_coverage_id.tolist() == result.spnode_age_id.tolist()
    for branch_id in result.index:
        assert result.loc[branch_id, "spnode_coverage_id"] == branch_id
    assert species.write() == species_before and gene.write() == gene_before
    assert all(not hasattr(node, "branch_id") for node in species.traverse())


def test_species_mapping_ids_mark_missing_species():
    with pytest.warns(RuntimeWarning, match="missing"):
        result = kfog.node_gene2species("(A_x_g:1,B_x_g:1);", "(A_x:1,C_x:1);", is_ultrametric=True)
    assert result.spnode_coverage_id.isna().sum() == 2
    assert result.spnode_age_id.isna().sum() == 2
    assert str(result.spnode_coverage_id.dtype) == "Int64"


def ou_files(tmp_path, trait, regime_id=1):
    tree, regimes, means = (tmp_path / name for name in ["tree.nwk", "regimes.tsv", "means.tsv"])
    tree.write_text("(A:1,B:1)Root;")
    regimes.write_text(f"node_name\tregime\nRoot\t0\nA\t{regime_id}\nB\t\n")
    means.write_text(f"node_name\tparam\tregime\t{trait}\nRoot\tmu\t0\t1\nA\tmu\t{regime_id}\t3\nRoot\talpha\t\t1\n")
    return regimes, means, tree


@pytest.mark.parametrize("trait", ["complementarity", "complementarity_parent", "complementarity_sisters"])
def test_ou_rejects_derived_column_collisions(tmp_path, trait):
    with pytest.raises(ValueError, match="reserved OU statistic"):
        kfog.ou2table(*ou_files(tmp_path, trait))


@pytest.mark.parametrize("regime_id", [2**53 + 1, 2**63 - 1])
def test_regime_ids_are_lossless_with_missing_rows(tmp_path, regime_id):
    files = ou_files(tmp_path, "trait", regime_id)
    summary = kfog.regime2tree(files[1])
    assert summary["num_regime"] == regime_id + 1
    result = kfog.ou2table(*files)
    assert result.columns.is_unique
    assert result.loc[result.is_shift == 1, "regime"].iloc[0] == regime_id
    assert result.loc[result.is_shift == 1, "mu_trait"].iloc[0] == 3


@pytest.mark.parametrize("token", ["1.0", "1e0"])
def test_regime_integral_numeric_notation_remains_supported(tmp_path, token):
    files = ou_files(tmp_path, "trait", token)
    assert kfog.regime2tree(files[1])["num_regime"] == 2
    assert kfog.ou2table(*files).regime.max() == 1


@pytest.mark.parametrize("colors", ["red", MappingProxyType({"a": "red", "b": "blue"})])
def test_histogram_uses_string_and_mapping_colors(colors):
    data = pd.DataFrame({"v": [1, 2, 3, 4], "group": ["a", "a", "b", "b"]})
    ax = kfplot.hist_boxplot("v", "group", data, colors=colors)
    try:
        expected = ["red", "red"] if isinstance(colors, str) else ["red", "blue"]
        assert [patch.get_edgecolor() for patch in ax.patches] == [to_rgba(color, 0.9) for color in expected]
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("scale", [1.0, 1e-309, 1e307])
def test_root_theta_is_invariant_to_branch_scale(scale):
    tree = PhyloTree(f"(A:{scale},B:{2 * scale});")
    thetas = {"A": [{"theta": 0.2}] * 3, "B": [{"theta": 0.8}] * 3}
    result = kfseq.weighted_mean_root_thetas(thetas, tree, "F3X4")
    np.testing.assert_allclose([row["theta"] for row in result], [0.4] * 3)


def test_internal_name_transfer_checks_single_tip_identity():
    with pytest.raises(ValueError, match="identical tips"):
        kfphylo.transfer_internal_node_names("A;", "B;")
    assert kfphylo.transfer_internal_node_names("A;", "A;").name == "A"
