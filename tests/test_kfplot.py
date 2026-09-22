"""Plot input boundaries and rendering contracts; numerical fits live in test_plot_statistics."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kftools import kfplot


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_histogram_colors_and_categories():
    data = pd.DataFrame({"v": [0.0, 1.0, 2.0, 3.0], "g": ["A", "A", "B", "B"]})
    colors = ["red", "blue"]
    ax = kfplot.hist_boxplot("v", "g", data, colors=colors, xlim=[0, 4])
    assert colors == ["red", "blue"]
    assert ax.lines[0].get_color() == "red"
    assert {"A", "B"} <= {tick.get_text() for tick in ax.get_yticklabels()}
    # A partial mapping must still draw the category without an explicit color.
    partial = kfplot.hist_boxplot("v", "g", data, colors={"A": "red"}, xlim=[0, 4])
    assert {"A", "B"} <= {tick.get_text() for tick in partial.get_yticklabels()}


@pytest.mark.parametrize(
    ("x", "y", "options", "error"),
    [
        ([np.nan], [np.nan], {}, "no finite data points"),
        ([1, 2], [1, 2, 3], {}, "same shape"),
        ([1, 2], [1, 2], {"num_bin": 0}, "positive integer"),
        ([1, 2], [1, 2], {"vmin": 2, "vmax": 1}, "vmin"),
        ([1, 2], [1, 2], {"plot_range": [2, 1, 0, 1]}, "xmin <= xmax"),
    ],
)
def test_density_rejects_unusable_data_or_ranges(x, y, options, error):
    with pytest.raises(ValueError, match=error):
        kfplot.density_scatter(x, y, cbar=False, **options)


def test_density_coerces_and_filters_dataframe_values():
    ax = kfplot.density_scatter("x", "y", pd.DataFrame({"x": ["1.0", "inf"], "y": [1.0, 2.0]}), cbar=False)
    assert ax is not None  # Also exercises the single surviving observation without warnings.


@pytest.mark.parametrize("show_cor_p", [False, True])
def test_density_single_point_has_no_singular_axis_or_correlation_warning(show_cor_p):
    kfplot.density_scatter([1.0], [1.0], cbar=False, show_cor_p=show_cor_p)


def test_density_poisson_log_link_accepts_zero_response():
    # Warning-strict pytest catches invalid log transforms and perfect-fit warnings.
    kfplot.density_scatter([1, 2, 3, 4], [0, 1, 2, 3], reg_family=sm.families.Poisson(), cbar=False)


@pytest.mark.parametrize(
    ("groups", "error"),
    [
        ([[1], [2]], "hashable"),
        ([("A",), ("B",)], "scalar"),
        ([1, True], "mix bool"),
        ([1, "2"], "mix string"),
        ([0.1, np.inf], "non-finite"),
    ],
)
def test_stacked_categories_cannot_silently_merge_or_disappear(groups, error):
    data = pd.DataFrame({"g": groups, "value": [1, 2]})
    before = plt.get_fignums()
    with pytest.raises(ValueError, match=error):
        kfplot.stacked_barplot("g", ["value"], data, ["red"], ax=None)
    assert plt.get_fignums() == before


def test_stacked_axes_and_duplicate_components_are_rejected():
    data = pd.DataFrame({"g": ["A"], "value": [1]})
    with pytest.raises(ValueError, match="Exactly one"):
        kfplot.stacked_barplot("g", "value", data, ["red"], ax=None)
    with pytest.raises(ValueError, match="duplicate"):
        kfplot.stacked_barplot("g", ["value", "value"], data, ["red"], ax=None)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"alpha": 2}, "between 0 and 1"),
        ({"box_step": -0.1}, "positive"),
        ({"xlim": [2, 1]}, "xmin <= xmax"),
        ({"bins": [2, 1]}, "strictly increasing"),
        ({"bins": [1, np.inf]}, "finite numeric bin-edge"),
        ({"colors": {"A": "red", "Z": "blue"}}, "colors contains categories"),
    ],
)
def test_histogram_rejects_misleading_plot_options(options, error):
    with pytest.raises(ValueError, match=error):
        kfplot.hist_boxplot("x", "g", pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}), **options)


def test_histogram_rejects_empty_categories_and_cleans_up():
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="has no non-NaN values"):
        kfplot.hist_boxplot("x", "g", pd.DataFrame({"x": [1.0, np.nan], "g": ["A", "B"]}))
    assert plt.get_fignums() == before


def test_histogram_boolean_values_do_not_warn():
    kfplot.hist_boxplot("x", "g", pd.DataFrame({"x": [True, False, True], "g": ["A", "A", "B"]}))


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"method": "bad"}, "must be either"),
        ({"stats": ["N", "bad"]}, "unsupported entries"),
        ({"textxy": [0.1]}, "exactly"),
    ],
)
def test_annotation_rejects_unsupported_options(options, error):
    with pytest.raises(ValueError, match=error):
        kfplot.ols_annotations([1, 2, 3], [2, 1, 4], **options)


def test_annotation_preserves_row_alignment_with_nondefault_index():
    data = pd.DataFrame({"x": [3.0, 1.0, 2.0], "y": [6.0, 2.0, 4.0]}, index=[10, 30, 20])
    ax = kfplot.ols_annotations("x", "y", data, method="ols", stats="N")
    assert ax.texts[0].get_text().strip() == "N = 3"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [2, 6])


@pytest.mark.parametrize("method", ["ols", "quantreg"])
def test_annotation_constant_response_has_no_inferential_statistics(method):
    ax = kfplot.ols_annotations([2, 0, 1], [1, 1, 1], method=method, stats=["slope", "slope_p", "rsquared"])
    assert "slope = 0.00" in ax.texts[0].get_text()
    assert "P = NaN" in ax.texts[0].get_text()
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 1])


def test_annotation_two_point_fit_has_no_residual_degrees_of_freedom():
    ax = kfplot.ols_annotations([0, 2], [0, 1], method="ols", stats=["rsquared", "slope_p"])
    assert "R2 = 1.00" in ax.texts[0].get_text()
    assert "P = NaN" in ax.texts[0].get_text()
