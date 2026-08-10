import warnings
from decimal import Decimal
from typing import Any

import matplotlib.pyplot
import numpy as np
import pandas as pd
import scipy.stats as stats
from pandas.api.types import is_scalar

from ._validation import is_hashable, validate_boolean_flag

_DEFAULT_TEXTXY = object()


def _pearsonr_fast(xval, yval):
    xarr = np.asarray(xval, dtype=float)
    yarr = np.asarray(yval, dtype=float)
    if xarr.shape[0] != yarr.shape[0]:
        raise ValueError("xval and yval must have the same length")
    n = xarr.shape[0]
    if n < 2:
        return np.nan, np.nan

    x_center = xarr - xarr.mean()
    y_center = yarr - yarr.mean()
    x_norm = float(np.sqrt(np.dot(x_center, x_center)))
    y_norm = float(np.sqrt(np.dot(y_center, y_center)))
    if (x_norm == 0.0) or (y_norm == 0.0):
        return np.nan, np.nan

    r = float(np.dot(x_center, y_center) / (x_norm * y_norm))
    r = float(np.clip(r, -1.0, 1.0))
    if n < 3:
        return r, np.nan
    if abs(r) == 1.0:
        return r, 0.0
    t_stat = r * np.sqrt((n - 2) / (1 - (r * r)))
    pval = float(2 * stats.t.sf(abs(t_stat), n - 2))
    return r, pval


def _spearmanr_fast(xval, yval):
    rank_x = stats.rankdata(xval)
    rank_y = stats.rankdata(yval)
    return _pearsonr_fast(rank_x, rank_y)


def _resolve_series_color(colors, idx):
    if isinstance(colors, str):
        return colors
    if isinstance(colors, np.ndarray):
        if colors.ndim == 0:
            return colors.item()
        if len(colors) > 0:
            return colors[idx % len(colors)]
    if isinstance(colors, (list, tuple)) and (len(colors) > 0):
        return colors[idx % len(colors)]
    return f"C{idx}"


def _coerce_numeric_columns(data, columns, argument_name):
    converted = {}
    for col in columns:
        numeric_series = pd.to_numeric(data[col], errors="coerce")
        non_missing_original = data[col].notna().to_numpy()
        invalid_numeric_mask = non_missing_original & numeric_series.isna().to_numpy()
        if invalid_numeric_mask.any():
            invalid_values = sorted(set(data.loc[invalid_numeric_mask, col].astype(str)))
            raise ValueError(
                f"{argument_name} columns must contain numeric values; invalid values in '{col}': {invalid_values}"
            )
        numeric_values = numeric_series.to_numpy(dtype=float, copy=False)
        non_finite_mask = non_missing_original & (~np.isfinite(numeric_values))
        if non_finite_mask.any():
            invalid_values = sorted(set(data.loc[non_finite_mask, col].astype(str)))
            raise ValueError(
                f"{argument_name} columns must contain finite numeric values; "
                f"invalid values in '{col}': {invalid_values}"
            )
        converted[col] = numeric_series
    return pd.DataFrame(converted)


def _limited_value_examples(values, predicate):
    return [str(value) for value in values if predicate(value)][:5]


def _is_non_bool_numeric(value):
    return (not isinstance(value, (bool, np.bool_))) and isinstance(value, (int, float, np.integer, np.floating))


def _validate_hashable_column_values(data, column_name, argument_name):
    values = data[column_name].dropna().to_list()
    unhashable_examples = _limited_value_examples(values, lambda value: not is_hashable(value))
    if len(unhashable_examples) > 0:
        raise ValueError(
            f"{argument_name} column '{column_name}' must contain hashable values; "
            f"invalid examples: {unhashable_examples}"
        )
    non_scalar_examples = _limited_value_examples(values, lambda value: not is_scalar(value))
    if non_scalar_examples:
        raise ValueError(
            f"{argument_name} column '{column_name}' must contain scalar values; "
            f"invalid examples: {non_scalar_examples}"
        )
    complex_examples = _limited_value_examples(values, lambda value: isinstance(value, (complex, np.complexfloating)))
    if complex_examples:
        raise ValueError(
            f"{argument_name} column '{column_name}' must not contain complex values; "
            f"invalid examples: {complex_examples}"
        )
    bool_values = [value for value in values if isinstance(value, (bool, np.bool_))]
    numeric_values = [value for value in values if _is_non_bool_numeric(value)]
    if bool_values and numeric_values:
        bool_numeric_collision_examples = [str(value) for value in (bool_values + numeric_values)[:5]]
        raise ValueError(
            f"{argument_name} column '{column_name}' must not mix bool and numeric non-bool values; "
            f"invalid examples: {bool_numeric_collision_examples}"
        )
    non_finite_numeric_examples = _limited_value_examples(numeric_values, lambda value: not np.isfinite(float(value)))
    if non_finite_numeric_examples:
        raise ValueError(
            f"{argument_name} column '{column_name}' must not contain non-finite numeric values; "
            f"invalid examples: {non_finite_numeric_examples}"
        )


def _normalize_plot_category_labels(category_labels, argument_name):
    labels = list(category_labels)
    has_string_label = any(isinstance(label, str) for label in labels)
    has_non_string_label = any(not isinstance(label, str) for label in labels)
    if has_string_label and has_non_string_label:
        raise ValueError(f"{argument_name} grouping categories must not mix string and non-string values")
    return labels


def _validate_stacked_column_list(columns, axis_name):
    if len(columns) == 0:
        raise ValueError(f"{axis_name} list must contain at least one column")
    invalid_columns = [col for col in columns if (not isinstance(col, str)) or (col.strip() == "")]
    if invalid_columns:
        raise ValueError(
            f"{axis_name} list must contain non-empty string column names; invalid entries: {invalid_columns}"
        )
    if len(set(columns)) != len(columns):
        raise ValueError(f"{axis_name} list must not contain duplicate column names")


def _validate_stacked_axes(x, y, data):
    x_is_str = isinstance(x, str)
    y_is_str = isinstance(y, str)
    x_is_list = isinstance(x, list)
    y_is_list = isinstance(y, list)
    if x_is_str == y_is_str:
        raise ValueError("Exactly one of x and y must be a list, and the other must be a string column")
    if x_is_list == y_is_list:
        raise ValueError("Exactly one of x and y must be a list")
    if (x_is_str and (not y_is_list)) or (y_is_str and (not x_is_list)):
        raise ValueError("x/y types are invalid; expected one list and one string")
    if x_is_str and (x.strip() == ""):
        raise ValueError("x column name must be a non-empty string")
    if y_is_str and (y.strip() == ""):
        raise ValueError("y column name must be a non-empty string")
    if x_is_list:
        _validate_stacked_column_list(x, "x")
    if y_is_list:
        _validate_stacked_column_list(y, "y")
    required_cols = list(x) + [y] if x_is_list else [x] + list(y)
    missing_cols = [col for col in required_cols if col not in data.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Columns not found in data: {missing_cols}")
    category_axis = y if x_is_list else x
    _validate_hashable_column_values(
        data=data,
        column_name=category_axis,
        argument_name="y" if x_is_list else "x",
    )
    return x_is_list, y_is_list


def _stacked_frames(x, y, data, x_is_list, y_is_list):
    dfs = {
        "x": _coerce_numeric_columns(data, x, "x") if x_is_list else pd.DataFrame(data.loc[:, x]),
        "y": _coerce_numeric_columns(data, y, "y") if y_is_list else pd.DataFrame(data.loc[:, y]),
    }
    if x_is_list:
        dfs["x"] = dfs["x"].cumsum(axis=1)
    if y_is_list:
        dfs["y"] = dfs["y"].cumsum(axis=1)
    return dfs, pd.concat([dfs["x"], dfs["y"]], axis=1)


def _draw_stacked_bars(ax, x, y, colors, dfs, df, x_is_list):
    if x_is_list:
        columns, category, draw = dfs["x"].columns, y, ax.barh
        category_argument = "y"
    else:
        columns, category, draw = dfs["y"].columns, x, ax.bar
        category_argument = "x"
    grouped = df.groupby(category, sort=False)[list(columns)].mean()
    for i in reversed(range(len(columns))):
        value_column = columns[i]
        grouped_values = grouped[value_column].dropna()
        draw(
            _normalize_plot_category_labels(grouped_values.index, argument_name=category_argument),
            grouped_values.values,
            color=_resolve_series_color(colors, i),
            linewidth=0,
        )


def stacked_barplot(x: Any, y: Any, data: Any, colors: Any, ax: Any) -> Any:
    """Draw horizontally or vertically stacked means grouped by a category."""
    if not hasattr(data, "columns"):
        raise ValueError("data must be a pandas DataFrame-like object with columns")
    x_is_list, y_is_list = _validate_stacked_axes(x, y, data)
    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True

    try:
        dfs, combined = _stacked_frames(x, y, data, x_is_list, y_is_list)
        _draw_stacked_bars(ax, x, y, colors, dfs, combined, x_is_list)
    except Exception:
        if created_internal_ax:
            matplotlib.pyplot.close(ax.figure)
        raise
    return ax


def _validate_density_options(cor, diag, hue_log, show_cor_p, return_ims, cbar, num_bin, vmin, vmax):
    flags = [
        validate_boolean_flag(value, name)
        for name, value in [
            ("cor", cor),
            ("diag", diag),
            ("hue_log", hue_log),
            ("show_cor_p", show_cor_p),
            ("return_ims", return_ims),
            ("cbar", cbar),
        ]
    ]
    if isinstance(num_bin, bool) or (not isinstance(num_bin, (int, np.integer))) or (num_bin <= 0):
        raise ValueError("num_bin must be a positive integer")
    for bound_name, bound_value in [("vmin", vmin), ("vmax", vmax)]:
        if bound_value is None:
            continue
        if isinstance(bound_value, bool) or (not isinstance(bound_value, (int, float, np.integer, np.floating))):
            raise ValueError(f"{bound_name} must be None or a finite numeric value")
        if not np.isfinite(float(bound_value)):
            raise ValueError(f"{bound_name} must be None or a finite numeric value")
    if (vmin is not None) and (vmax is not None) and (float(vmin) > float(vmax)):
        raise ValueError("vmin must be less than or equal to vmax")
    return (*flags, int(num_bin))


def _density_xy_values(x, y, df):
    if (df is not None) and (not hasattr(df, "columns")):
        raise ValueError("df must be a pandas DataFrame-like object with columns")
    if df is None:
        try:
            xval = np.asarray(x, dtype=float)
            yval = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("x and y must contain numeric values when df is None") from exc
        if xval.shape != yval.shape:
            raise ValueError("x and y must have the same shape when df is None")
        valid = np.isfinite(xval) & np.isfinite(yval)
        return "x", "y", xval[valid], yval[valid]
    if not isinstance(x, str) or not isinstance(y, str):
        raise ValueError("x and y must be string column names when df is provided")
    if (x not in df.columns) or (y not in df.columns):
        raise ValueError(f"df must include columns '{x}' and '{y}'")
    df_xy = df.loc[:, [x, y]].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    try:
        xval = pd.to_numeric(df_xy.iloc[:, 0], errors="raise").to_numpy(dtype=float)
        yval = pd.to_numeric(df_xy.iloc[:, 1], errors="raise").to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError(f"df columns '{x}' and '{y}' must contain numeric values") from exc
    valid = np.isfinite(xval) & np.isfinite(yval)
    return x, y, xval[valid], yval[valid]


def _fit_density_glm(xval, yval, reg_family):
    import statsmodels.api as sm

    if not hasattr(reg_family, "link"):
        raise ValueError("reg_family must be a statsmodels family object with a 'link' attribute")
    predictor = "_kftools_predictor"
    exog = sm.add_constant(pd.DataFrame({predictor: xval}), has_constant="add")
    try:
        result = sm.GLM(yval, exog, family=reg_family).fit()
    except Exception as exc:
        raise ValueError("GLM fit failed in density_scatter; check reg_family compatibility and input values") from exc
    endpoints = (float(xval.min()), float(xval.max()))
    x_predict = (
        np.array([endpoints[0]], dtype=float)
        if endpoints[0] == endpoints[1]
        else np.linspace(*endpoints, num=100, endpoint=True)
    )
    predict_exog = sm.add_constant(pd.DataFrame({predictor: x_predict}), has_constant="add")
    return x_predict, np.asarray(result.predict(predict_exog), dtype=float)


def _apply_density_log_link(xval, yval, y_predict):
    if np.any(y_predict <= 0):
        raise ValueError("GLM predicted non-positive values; cannot apply log transform in density_scatter")
    with np.errstate(divide="ignore", invalid="ignore"):
        transformed_y = np.log(yval)
    valid = np.isfinite(transformed_y)
    if not valid.any():
        raise ValueError("density_scatter has no finite values after log transformation")
    return xval[valid], transformed_y[valid], np.log(y_predict)


def _explicit_plot_range(plot_range):
    try:
        values = list(plot_range)
    except TypeError as exc:
        raise ValueError("plot_range must be either 'each'/'ceil' or [xmin, xmax, ymin, ymax]") from exc
    if len(values) != 4:
        raise ValueError("plot_range must be either 'each'/'ceil' or [xmin, xmax, ymin, ymax]")
    try:
        bounds = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("plot_range numeric bounds must be finite numbers") from exc
    if not np.isfinite(bounds).all():
        raise ValueError("plot_range numeric bounds must be finite numbers")
    xmin, xmax, ymin, ymax = bounds
    if xmin > xmax:
        raise ValueError("plot_range requires xmin <= xmax")
    if ymin > ymax:
        raise ValueError("plot_range requires ymin <= ymax")
    return bounds


def _density_plot_range(plot_range, xval, yval):
    if isinstance(plot_range, str):
        if plot_range not in ("each", "ceil"):
            raise ValueError("plot_range must be either 'each'/'ceil' or [xmin, xmax, ymin, ymax]")
        bounds = (np.floor(xval.min()), np.ceil(xval.max()), np.floor(yval.min()), np.ceil(yval.max()))
    else:
        bounds = _explicit_plot_range(plot_range)
    xmin, xmax, ymin, ymax = bounds
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5
    return xmin, xmax, ymin, ymax


def _density_histogram(xval, yval, bounds, num_bin, hue_log):
    xmin, xmax, ymin, ymax = bounds
    bins = [num_bin, num_bin]
    threshold = np.log2(3) if hue_log else 3
    xyrange = [[xmin, xmax], [ymin, ymax]]
    histogram, _, _ = np.histogram2d(xval, yval, range=xyrange, bins=bins)
    if hue_log:
        with np.errstate(divide="ignore"):
            histogram = np.log2(histogram)
    x_idx = np.floor((xval - xmin) * bins[0] / (xmax - xmin)).astype(int)
    y_idx = np.floor((yval - ymin) * bins[1] / (ymax - ymin)).astype(int)
    x_idx = np.minimum(x_idx, bins[0] - 1)
    y_idx = np.minimum(y_idx, bins[1] - 1)
    in_range = (x_idx >= 0) & (x_idx < bins[0]) & (y_idx >= 0) & (y_idx < bins[1])
    point_density = histogram[x_idx[in_range], y_idx[in_range]]
    low_density = point_density < threshold
    low_x, low_y = xval[in_range][low_density], yval[in_range][low_density]
    histogram[histogram < threshold] = np.nan
    return xyrange, histogram, low_x, low_y


def _add_density_colorbar(ax, image, hue_log):
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    colorbar = matplotlib.pyplot.colorbar(
        mappable=image,
        ax=ax,
        format=formatter,
    )
    colorbar.ax.tick_params(axis="y", which="major", direction="out", length=3, width=1, pad=2)
    if hue_log:
        colorbar.ax.text(0.5, 1.001, "log$_2$ count", ha="center", va="bottom")
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    colorbar.update_ticks()


def _density_correlation_title(xval, yval, show_cor_p):
    spearman, spearman_p = _spearmanr_fast(xval=xval, yval=yval)
    pearson, pearson_p = _pearsonr_fast(xval=xval, yval=yval)
    if not show_cor_p:
        return f"ρ={spearman:.2f}, r={pearson:.2f}"
    spearman_p_text = "0" if spearman_p == 0 else f"{spearman_p:.2e}"
    pearson_p_text = "0" if pearson_p == 0 else f"{pearson_p:.2e}"
    return f"ρ={spearman:.2f} p={spearman_p_text}, r={pearson:.2f} p={pearson_p_text}"


def _draw_density_scatter(ax, plot_data, labels, options, correlation_values, glm_prediction):
    xyrange, histogram, low_x, low_y, bounds = plot_data
    xlabel, ylabel = labels
    cmap, vmin, vmax, cbar, hue_log, cor, show_cor_p, diag, plot_range = options
    image = ax.imshow(
        np.flipud(histogram.T),
        cmap=cmap,
        extent=np.array(xyrange).flatten(),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
        origin="upper",
        aspect="auto",
    )
    ax.plot(low_x, low_y, ".", color="darkblue")
    if cbar:
        _add_density_colorbar(ax, image, hue_log)
    if glm_prediction is not None:
        ax.plot(*glm_prediction, "-", color="red", lw=2)
    xmin, xmax, ymin, ymax = bounds
    if isinstance(plot_range, str) and (plot_range == "ceil"):
        xymin, xymax = min(xmin, ymin), max(xmax, ymax)
        ax.set_xlim(xymin, xymax)
        ax.set_ylim(xymin, xymax)
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1, pad=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if cor:
        correlation_x, correlation_y = correlation_values
        ax.set_title(
            _density_correlation_title(correlation_x, correlation_y, show_cor_p),
            fontsize=matplotlib.rcParams["font.size"],
        )
    if diag:
        diagonal = np.asarray([min(xmin, ymin), max(xmax, ymax)], dtype=float)
        ax.plot(diagonal, diagonal, "-", color="black", lw=1)
    return image


def density_scatter(
    x: Any,
    y: Any,
    df: Any = None,
    ax: Any = None,
    cor: bool = True,
    diag: bool = False,
    reg_family: Any = None,
    hue_log: bool = False,
    show_cor_p: bool = True,
    plot_range: Any = "each",
    return_ims: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: bool = True,
    cmap: Any = "jet",
    num_bin: int = 20,
) -> Any:
    """Draw a density-colored scatter plot with optional correlation and GLM."""
    cor, diag, hue_log, show_cor_p, return_ims, cbar, num_bin = _validate_density_options(
        cor, diag, hue_log, show_cor_p, return_ims, cbar, num_bin, vmin, vmax
    )
    xlabel, ylabel, xval, yval = _density_xy_values(x, y, df)
    if xval.size == 0:
        raise ValueError("density_scatter received no finite data points")
    glm_prediction = None
    if reg_family is not None:
        x_predict, y_predict = _fit_density_glm(xval, yval, reg_family)
        if reg_family.link.__class__.__name__.lower() == "log":
            xval, yval, y_predict = _apply_density_log_link(xval, yval, y_predict)
        glm_prediction = (x_predict, y_predict)
    bounds = _density_plot_range(plot_range, xval, yval)
    xyrange, hh, xdat1, ydat1 = _density_histogram(xval, yval, bounds, num_bin, hue_log)

    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True

    try:
        ims = _draw_density_scatter(
            ax,
            (xyrange, hh, xdat1, ydat1, bounds),
            (xlabel, ylabel),
            (cmap, vmin, vmax, cbar, hue_log, cor, show_cor_p, diag, plot_range),
            (xval, yval),
            glm_prediction,
        )
    except Exception:
        if created_internal_ax:
            matplotlib.pyplot.close(ax.figure)
        raise

    return ims if return_ims else ax


def _validate_hist_inputs(df, x, category, alpha, box_step):
    if df is None:
        df = pd.DataFrame()
    if not hasattr(df, "columns"):
        raise ValueError("df must be a pandas DataFrame-like object with columns")
    if (not isinstance(x, str)) or (x.strip() == ""):
        raise ValueError("x must be a non-empty string column name")
    if (not isinstance(category, str)) or (category.strip() == ""):
        raise ValueError("category must be a non-empty string column name")
    if category not in df.columns:
        raise ValueError(f"category column '{category}' was not found in df")
    if x not in df.columns:
        raise ValueError(f"x column '{x}' was not found in df")
    alpha = _finite_plot_number(alpha, "alpha must be a finite numeric value between 0 and 1")
    if (alpha < 0) or (alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    box_step = _finite_plot_number(box_step, "box_step must be a positive finite numeric value")
    if box_step <= 0:
        raise ValueError("box_step must be a positive finite numeric value")
    return df, alpha, box_step


def _finite_plot_number(value, error_message):
    if (
        isinstance(value, bool)
        or (not isinstance(value, (int, float, np.integer, np.floating)))
        or (not np.isfinite(float(value)))
    ):
        raise ValueError(error_message)
    return float(value)


def _hist_numeric_frame(df, x, category):
    _validate_hashable_column_values(data=df, column_name=category, argument_name="category")
    x_numeric = pd.to_numeric(df[x], errors="coerce")
    invalid_x_mask = df[x].notna() & x_numeric.isna()
    if invalid_x_mask.any():
        invalid_values = sorted(set(df.loc[invalid_x_mask, x].astype(str)))
        raise ValueError(f"x column '{x}' must contain numeric values; invalid values: {invalid_values}")
    non_finite_x_mask = x_numeric.notna() & np.isinf(x_numeric)
    if non_finite_x_mask.any():
        invalid_values = sorted(set(df.loc[non_finite_x_mask, x].astype(str)))
        raise ValueError(f"x column '{x}' must contain finite numeric values; invalid values: {invalid_values}")
    out = df.copy()
    out[x] = x_numeric.astype(float)
    return out


def _hist_xlim(xlim, df, x):
    try:
        values = list(xlim)
    except TypeError as exc:
        raise ValueError("xlim must be empty or contain exactly [xmin, xmax]") from exc
    if len(values) not in (0, 2):
        raise ValueError("xlim must be empty or contain exactly [xmin, xmax]")
    if not values:
        x_values = df[x].dropna()
        if x_values.empty:
            raise ValueError("hist_boxplot requires at least one non-NaN value in the x column")
        values = [float(x_values.min()), float(x_values.max())]
    try:
        values = [float(values[0]), float(values[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError("xlim bounds must be finite numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("xlim bounds must be finite numeric values")
    if values[0] > values[1]:
        raise ValueError("xlim requires xmin <= xmax")
    if values[0] == values[1]:
        values = [values[0] - 0.5, values[1] + 0.5]
    return values


def _hist_bins(bins, xlim):
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins <= 0:
            raise ValueError("bins must be a positive integer when scalar")
        bins = np.linspace(xlim[0], xlim[1], num=(int(bins) + 1), endpoint=True)
    elif isinstance(bins, str):
        raise ValueError("bins must be empty, a positive integer, or an array-like of bin edges")
    else:
        try:
            bins = list(bins)
        except TypeError as exc:
            raise ValueError("bins must be empty, a positive integer, or an array-like of bin edges") from exc
        if not bins:
            span = xlim[1] - xlim[0]
            bins = np.arange(xlim[0] - (span / 50), xlim[1] + (span / 50), span / 100)
    try:
        bins = np.asarray(bins, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("bins must contain finite numeric bin-edge values") from exc
    _validate_hist_bin_edges(bins)
    return bins


def _validate_hist_bin_edges(bins):
    if bins.shape[0] < 2:
        raise ValueError("bins must define at least 2 bin-edge values")
    if not np.isfinite(bins).all():
        raise ValueError("bins must contain finite numeric bin-edge values")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be strictly increasing")


def _hist_categories_and_colors(df, category, colors):
    category_values = list(df[category].dropna().drop_duplicates())
    if not category_values:
        raise ValueError(f"category column '{category}' must contain at least one non-NaN value")
    if isinstance(colors, dict) and colors:
        observed = df[category].dropna().tolist()
        unknown = [category_value for category_value in colors if category_value not in observed]
        if unknown:
            raise ValueError(f"colors contains categories that are not present in df[{category!r}]: {unknown}")
    return category_values


def _hist_category_color(colors, category_value, index):
    if isinstance(colors, dict):
        color = colors.get(category_value)
    elif isinstance(colors, np.ndarray):
        color = colors.item() if colors.ndim == 0 else colors[index % len(colors)] if len(colors) else None
    elif isinstance(colors, (list, tuple)):
        color = colors[index % len(colors)] if colors else None
    else:
        color = f"C{index}"
    return f"C{index}" if color is None else color


def _draw_hist_box_categories(ax, df, x, category, category_values, colors, bins, alpha, box_step):
    box_position = 1 + (box_step * len(category_values))
    yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grouped_values = {key: group[x].dropna() for key, group in df.groupby(category, sort=False)}
    for index, category_value in enumerate(category_values):
        color = _hist_category_color(colors, category_value, index)
        values = grouped_values.get(category_value, pd.Series(dtype=float)).to_numpy(copy=False)
        if values.size == 0:
            raise ValueError(f"Category '{category_value}' has no non-NaN values in column '{x}'")
        ax.hist(
            values,
            bins=bins,
            cumulative=True,
            histtype="step",
            lw=1,
            alpha=alpha,
            density=True,
            color=color,
            label=category_value,
        )
        try:
            box = ax.boxplot(
                values,
                positions=[box_position],
                orientation="horizontal",
                showfliers=False,
                widths=[0.1],
            )
        except TypeError:
            box = ax.boxplot(values, positions=[box_position], vert=False, showfliers=False, widths=[0.1])
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            matplotlib.pyplot.setp(box[element], color=color, linestyle="solid")
        yticks.append(box_position)
        box_position -= box_step
    return yticks


def hist_boxplot(
    x: str = "",
    category: str = "",
    df: Any = None,
    colors: Any = None,
    xlim: Any = None,
    bins: Any = None,
    alpha: float = 0.9,
    box_step: float = 0.15,
    ax: Any = None,
) -> Any:
    """Draw cumulative histograms and compact box plots by category."""
    df, alpha, box_step = _validate_hist_inputs(df, x, category, alpha, box_step)
    colors = {} if colors is None else colors
    xlim = [] if xlim is None else xlim
    bins = [] if bins is None else bins
    df = _hist_numeric_frame(df, x, category)
    xlim = _hist_xlim(xlim, df, x)
    bins = _hist_bins(bins, xlim)
    category_values = _hist_categories_and_colors(df, category, colors)
    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True

    try:
        yticks = _draw_hist_box_categories(ax, df, x, category, category_values, colors, bins, alpha, box_step)
        ax.set_xlabel(x)
        ax.set_ylabel("Cumulative frequency")
        ax.set_xlim(np.mean([xlim[0], min(bins)]), np.mean([xlim[1], max(bins)]))
        ax.set_ylim(-0.02, 1.1 + (box_step * len(category_values)))
        ax.set_yticks(yticks)
        yticklabels = [y for y in yticks if y <= 1] + category_values
        ax.set_yticklabels(yticklabels)
    except Exception:
        if created_internal_ax:
            matplotlib.pyplot.close(ax.figure)
        raise
    return ax


def _normalize_annotation_stats(stats):
    if stats is None:
        stats = ["N", "slope", "slope_p"]
    elif isinstance(stats, str):
        stats = [stats]
    else:
        try:
            stats = list(stats)
        except TypeError as exc:
            raise ValueError("stats must be a string or a sequence of statistic names") from exc
    allowed = {"N", "slope", "slope_p", "rsquared", "rsquared_p"}
    invalid = [stat for stat in stats if (not isinstance(stat, str)) or (stat not in allowed)]
    if invalid:
        raise ValueError(f"stats contains unsupported entries: {invalid}")
    return stats


def _annotation_text_coordinates(textxy):
    if textxy is _DEFAULT_TEXTXY:
        return [0.05, 0.95]
    try:
        values = list(textxy)
    except TypeError as exc:
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates") from exc
    if len(values) != 2:
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates")
    try:
        values = [float(values[0]), float(values[1])]
    except (TypeError, ValueError) as exc:
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates") from exc
    if not np.isfinite(values).all():
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates")
    return values


def _annotation_numeric_data(x, y, data):
    if data is None:
        data, x, y = pd.DataFrame({"X": x, "Y": y}), "X", "Y"
    elif not hasattr(data, "columns"):
        raise ValueError("data must be a pandas DataFrame-like object with columns")
    if not isinstance(x, str) or not isinstance(y, str):
        raise ValueError("x and y must be string column names")
    if (x not in data.columns) or (y not in data.columns):
        raise ValueError(f"data must include columns '{x}' and '{y}'")
    if data.shape[0] < 2:
        raise ValueError("ols_annotations requires at least 2 rows")
    try:
        x_numeric = pd.to_numeric(data[x], errors="raise").astype(float)
        y_numeric = pd.to_numeric(data[y], errors="raise").astype(float)
    except Exception as exc:
        raise ValueError("ols_annotations requires numeric x and y values") from exc
    if (not np.isfinite(x_numeric).all()) or (not np.isfinite(y_numeric).all()):
        raise ValueError("ols_annotations requires finite numeric values in x and y")
    out = data.copy()
    out[x], out[y] = x_numeric, y_numeric
    return x, y, out.sort_values(x)


def _fit_annotation_model(data, x, y, method, y_has_variation):
    import statsmodels.api as sm

    exog = sm.add_constant(data.loc[:, [x]], has_constant="add")
    response = data.loc[:, y]
    if method == "ols":
        try:
            return sm.OLS(response, exog).fit()
        except Exception as exc:
            raise ValueError("ols fit failed in ols_annotations") from exc
    if method != "quantreg":
        raise ValueError("method must be either 'ols' or 'quantreg'")
    if not y_has_variation:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                return sm.QuantReg(response, exog).fit(q=0.5)
    except Exception:
        return None


def _result_value(result_values, key):
    try:
        value = float(result_values[key])
    except (KeyError, TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _ols_r_squared(result, y_has_variation):
    if (float(result.df_resid) <= 0) or (not y_has_variation):
        return np.nan, np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (result.rsquared_adj, result.f_pvalue)
    return tuple(_finite_result_scalar(value) for value in values)


def _finite_result_scalar(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _quantreg_r_squared(result):
    with np.errstate(divide="ignore", invalid="ignore"):
        value = getattr(result, "prsquared", np.nan)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def _annotation_statistics(result, x, method, requested_stats, y_has_variation):
    values = {"slope": np.nan, "slope_p": np.nan, "rsquared": np.nan, "rsquared_p": np.nan}
    if result is None:
        return values
    if {"slope", "slope_p"} & set(requested_stats):
        values["slope"] = _result_value(result.params, x)
        values["slope_p"] = _result_value(result.pvalues, x)
    if not ({"rsquared", "rsquared_p"} & set(requested_stats)):
        return values
    if method == "ols":
        values["rsquared"], values["rsquared_p"] = _ols_r_squared(result, y_has_variation)
    else:
        values["rsquared"] = _quantreg_r_squared(result)
    return values


def _annotation_text(stats, sample_size, values):
    formats = {
        "N": lambda: f"N = {sample_size:,}\n",
        "slope": lambda: f"slope = {Decimal(values['slope']):.2f}\n",
        "slope_p": lambda: f"P = {Decimal(values['slope_p']):.2E}\n",
        "rsquared": lambda: f"R2 = {Decimal(values['rsquared']):.2f}\n",
        "rsquared_p": lambda: f"P = {Decimal(values['rsquared_p']):.2E}\n",
    }
    return "".join(formats[stat]() for stat in stats)


def _draw_ols_annotation(ax, data, x, y_values, result, textxy, text, style):
    color, font_size, textva, textha = style
    ax.text(
        textxy[0],
        textxy[1],
        text,
        transform=ax.transAxes,
        va=textva,
        ha=textha,
        color=color,
        fontsize=font_size,
    )
    sample_size = data.shape[0]
    x_endpoints = data[x].to_numpy(copy=False)[[0, sample_size - 1]]
    if result is None:
        median = float(np.median(y_values))
        y_endpoints = np.asarray([median, median], dtype=float)
    else:
        y_endpoints = np.asarray(result.predict(), dtype=float)[[0, sample_size - 1]]
    ax.plot(x_endpoints, y_endpoints, color=color)


def ols_annotations(
    x: str,
    y: str,
    data: Any = None,
    ax: Any = None,
    color: Any = "black",
    font_size: float = 8,
    textxy: Any = _DEFAULT_TEXTXY,
    textva: str = "top",
    textha: str = "left",
    method: str = "quantreg",
    stats: Any = None,
) -> Any:
    """Fit a linear or quantile model and annotate its statistics on an axis."""
    stats = _normalize_annotation_stats(stats)
    textxy = _annotation_text_coordinates(textxy)
    x, y, data = _annotation_numeric_data(x, y, data)
    y_values = data[y].to_numpy(copy=False)
    y_has_variation = np.ptp(y_values) > 0
    res = _fit_annotation_model(data, x, y, method, y_has_variation)
    values = _annotation_statistics(res, x, method, stats, y_has_variation)
    text = _annotation_text(stats, data.shape[0], values)
    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True
    try:
        _draw_ols_annotation(
            ax,
            data,
            x,
            y_values,
            res,
            textxy,
            text,
            (color, font_size, textva, textha),
        )
    except Exception:
        if created_internal_ax:
            matplotlib.pyplot.close(ax.figure)
        raise
    return ax
