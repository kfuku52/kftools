"""Check plotted numerical values against independent calculations."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kftools import kfplot


@pytest.fixture
def ax():
    figure, axis = plt.subplots()
    yield axis
    plt.close(figure)


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_stacked_bars_average_components_before_stacking(ax, horizontal, dtype):
    data = pd.DataFrame({"group": ["A", "A"], "a": [10.0, 20.0], "b": [np.nan, 10.0]})
    data[["a", "b"]] = data[["a", "b"]].astype(dtype)
    before = data.copy(deep=True)
    x, y = (["a", "b"], "group") if horizontal else ("group", ["a", "b"])
    kfplot.stacked_barplot(x, y, data, ["red", "blue"], ax)
    sizes = [patch.get_width() if horizontal else patch.get_height() for patch in ax.patches]
    offsets = [patch.get_x() if horizontal else patch.get_y() for patch in ax.patches]
    np.testing.assert_allclose(sizes, [15, 10])
    np.testing.assert_allclose(offsets, [0, 15])
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("horizontal", [False, True])
def test_stacked_bars_separate_signs_and_ignore_unobserved_categories(ax, horizontal):
    data = pd.DataFrame(
        {
            "group": pd.Categorical(["B", "A"], categories=["A", "B", "C"]),
            "a": [3.0, -3.0],
            "b": [-2.0, 2.0],
            "c": [-1.0, 1.0],
        }
    )
    x, y = (["a", "b", "c"], "group") if horizontal else ("group", ["a", "b", "c"])
    kfplot.stacked_barplot(x, y, data, ["red", "blue", "green"], ax)
    sizes = [p.get_width() if horizontal else p.get_height() for p in ax.patches]
    offsets = [p.get_x() if horizontal else p.get_y() for p in ax.patches]
    np.testing.assert_allclose(sizes, [3, -3, -2, 2, -1, 1])
    np.testing.assert_allclose(offsets, [0, 0, 0, 0, -2, 2])


def test_ols_r_squared_and_adjusted_r_squared_are_distinct(ax):
    x = np.arange(1, 6, dtype=float)
    y = np.array([2, 1, 4, 3, 6], dtype=float)
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    prediction = y.mean() + slope * (x - x.mean())
    r_squared = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    adjusted = 1 - (1 - r_squared) * (len(x) - 1) / (len(x) - 2)
    kfplot.ols_annotations(x, y, ax=ax, method="ols", stats=["rsquared", "rsquared_adj"])
    assert ax.texts[0].get_text() == f"R2 = {r_squared:.2f}\nadj. R2 = {adjusted:.2f}\n"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), prediction[[0, -1]])


@pytest.mark.parametrize("method", ["ols", "quantreg"])
@pytest.mark.parametrize("name", ["const", "_kftools_predictor", "response", "x"])
def test_regression_is_invariant_to_predictor_column_name(ax, method, name):
    x, y = np.arange(1, 6, dtype=float), np.array([2, 1, 4, 3, 6], dtype=float)
    data = pd.DataFrame({name: x, "y": y})
    kfplot.ols_annotations(name, "y", data, ax=ax, method=method, stats=["slope", "slope_p", "rsquared"])
    text, line = ax.texts[0].get_text(), ax.lines[0].get_ydata().copy()
    kfplot.ols_annotations(x, y, ax=ax, method=method, stats=["slope", "slope_p", "rsquared"])
    assert text == ax.texts[1].get_text()
    np.testing.assert_allclose(line, ax.lines[1].get_ydata())
    if method == "ols":
        assert text.startswith("slope = 1.00\n")
    else:
        assert "pseudo R2 =" in text


@pytest.mark.parametrize("method", ["ols", "quantreg"])
def test_constant_predictor_has_no_identifiable_slope(ax, method):
    response = [0.0, 0.0, 9.0]
    kfplot.ols_annotations([3, 3, 3], response, ax=ax, method=method, stats=["slope", "slope_p", "rsquared"])
    assert "slope = NaN" in ax.texts[0].get_text()
    expected = np.mean(response) if method == "ols" else np.median(response)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [expected, expected])


def test_adjusted_r_squared_requires_ols(ax):
    with pytest.raises(ValueError, match="only available"):
        kfplot.ols_annotations([1, 2, 3], [2, 3, 4], ax=ax, method="quantreg", stats="rsquared_adj")
    assert not ax.lines and not ax.texts
