import matplotlib.pyplot
import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from decimal import Decimal
import warnings


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
    return f'C{idx}'


def _coerce_numeric_columns(data, columns, argument_name):
    converted = {}
    for col in columns:
        numeric_series = pd.to_numeric(data[col], errors='coerce')
        non_missing_original = data[col].notna().to_numpy()
        invalid_numeric_mask = non_missing_original & numeric_series.isna().to_numpy()
        if invalid_numeric_mask.any():
            invalid_values = sorted(set(data.loc[invalid_numeric_mask, col].astype(str)))
            raise ValueError(
                f"{argument_name} columns must contain numeric values; "
                f"invalid values in '{col}': {invalid_values}"
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


def _validate_hashable_column_values(data, column_name, argument_name):
    non_missing_values = data[column_name].dropna().to_list()
    unhashable_examples = []
    non_scalar_examples = []
    complex_examples = []
    bool_numeric_collision_examples = []
    non_finite_numeric_examples = []
    has_bool_value = False
    has_non_bool_numeric_value = False
    for column_value in non_missing_values:
        try:
            hash(column_value)
        except TypeError:
            unhashable_examples.append(str(column_value))
            if len(unhashable_examples) >= 5:
                break
            continue
        # pandas.Timestamp and similar extension scalars are valid grouping labels.
        if not is_scalar(column_value):
            non_scalar_examples.append(str(column_value))
            if len(non_scalar_examples) >= 5:
                break
        if isinstance(column_value, (complex, np.complexfloating)):
            complex_examples.append(str(column_value))
            if len(complex_examples) >= 5:
                break
        is_bool_value = isinstance(column_value, (bool, np.bool_))
        is_non_bool_numeric_value = (
            (not is_bool_value)
            and isinstance(column_value, (int, float, np.integer, np.floating))
        )
        if is_bool_value:
            has_bool_value = True
            if len(bool_numeric_collision_examples) < 5:
                bool_numeric_collision_examples.append(str(column_value))
        if is_non_bool_numeric_value:
            has_non_bool_numeric_value = True
            if len(bool_numeric_collision_examples) < 5:
                bool_numeric_collision_examples.append(str(column_value))
            if (len(non_finite_numeric_examples) < 5) and (not np.isfinite(float(column_value))):
                non_finite_numeric_examples.append(str(column_value))
    if len(unhashable_examples) > 0:
        raise ValueError(
            f"{argument_name} column '{column_name}' must contain hashable values; "
            f"invalid examples: {unhashable_examples}"
        )
    if len(non_scalar_examples) > 0:
        raise ValueError(
            f"{argument_name} column '{column_name}' must contain scalar values; "
            f"invalid examples: {non_scalar_examples}"
        )
    if len(complex_examples) > 0:
        raise ValueError(
            f"{argument_name} column '{column_name}' must not contain complex values; "
            f"invalid examples: {complex_examples}"
        )
    if has_bool_value and has_non_bool_numeric_value:
        raise ValueError(
            f"{argument_name} column '{column_name}' must not mix bool and numeric non-bool values; "
            f"invalid examples: {bool_numeric_collision_examples}"
        )
    if len(non_finite_numeric_examples) > 0:
        raise ValueError(
            f"{argument_name} column '{column_name}' must not contain non-finite numeric values; "
            f"invalid examples: {non_finite_numeric_examples}"
        )


def _normalize_plot_category_labels(category_labels, argument_name):
    labels = list(category_labels)
    has_string_label = any(isinstance(label, str) for label in labels)
    has_non_string_label = any(not isinstance(label, str) for label in labels)
    if has_string_label and has_non_string_label:
        raise ValueError(
            f"{argument_name} grouping categories must not mix string and non-string values"
        )
    return labels


def _validate_boolean_flag(flag_value, argument_name):
    if not isinstance(flag_value, (bool, np.bool_)):
        raise ValueError(f"{argument_name} must be a boolean value")
    return bool(flag_value)


def stacked_barplot(x, y, data, colors, ax):
    if not hasattr(data, "columns"):
        raise ValueError("data must be a pandas DataFrame-like object with columns")
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
    if x_is_list and (len(x) == 0):
        raise ValueError("x list must contain at least one column")
    if y_is_list and (len(y) == 0):
        raise ValueError("y list must contain at least one column")
    if x_is_str and (x.strip() == ""):
        raise ValueError("x column name must be a non-empty string")
    if y_is_str and (y.strip() == ""):
        raise ValueError("y column name must be a non-empty string")
    if x_is_list:
        invalid_x_columns = [col for col in x if (not isinstance(col, str)) or (col.strip() == "")]
        if len(invalid_x_columns) > 0:
            raise ValueError(f"x list must contain non-empty string column names; invalid entries: {invalid_x_columns}")
    if y_is_list:
        invalid_y_columns = [col for col in y if (not isinstance(col, str)) or (col.strip() == "")]
        if len(invalid_y_columns) > 0:
            raise ValueError(f"y list must contain non-empty string column names; invalid entries: {invalid_y_columns}")
    if x_is_list and (len(set(x)) != len(x)):
        raise ValueError("x list must not contain duplicate column names")
    if y_is_list and (len(set(y)) != len(y)):
        raise ValueError("y list must not contain duplicate column names")
    if x_is_list:
        required_cols = list(x) + [y]
    else:
        required_cols = [x] + list(y)
    missing_cols = [col for col in required_cols if col not in data.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Columns not found in data: {missing_cols}")
    if x_is_list:
        _validate_hashable_column_values(data=data, column_name=y, argument_name='y')
    else:
        _validate_hashable_column_values(data=data, column_name=x, argument_name='x')
    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True

    try:
        cols = {'x': x, 'y': y}
        dfs = {}
        if x_is_list:
            dfs['x'] = _coerce_numeric_columns(data=data, columns=x, argument_name='x')
        else:
            dfs['x'] = pd.DataFrame(data.loc[:, x])
        if y_is_list:
            dfs['y'] = _coerce_numeric_columns(data=data, columns=y, argument_name='y')
        else:
            dfs['y'] = pd.DataFrame(data.loc[:, y])
        for key, col in cols.items():
            if isinstance(col, list):
                dfs[key] = dfs[key].cumsum(axis=1)

        df = pd.concat([dfs['x'], dfs['y']], axis=1)
        if isinstance(cols['x'], list):
            ncol = dfs['x'].columns.shape[0]
            for i in reversed(range(ncol)):
                x_col = dfs['x'].columns[i]
                group_source = df.loc[:, [x_col, y]].copy()
                group_source[x_col] = pd.to_numeric(group_source[x_col], errors='coerce')
                grouped = group_source.groupby(y, sort=False)[x_col].mean().dropna()
                ax.barh(
                    _normalize_plot_category_labels(grouped.index, argument_name='y'),
                    grouped.values,
                    color=_resolve_series_color(colors, i),
                    linewidth=0,
                )
        if isinstance(cols['y'], list):
            ncol = dfs['y'].columns.shape[0]
            for i in reversed(range(ncol)):
                y_col = dfs['y'].columns[i]
                group_source = df.loc[:, [x, y_col]].copy()
                group_source[y_col] = pd.to_numeric(group_source[y_col], errors='coerce')
                grouped = group_source.groupby(x, sort=False)[y_col].mean().dropna()
                ax.bar(
                    _normalize_plot_category_labels(grouped.index, argument_name='x'),
                    grouped.values,
                    color=_resolve_series_color(colors, i),
                    linewidth=0,
                )
    except Exception:
        if created_internal_ax:
            matplotlib.pyplot.close(ax.figure)
        raise
    return ax


def density_scatter(x, y, df=None, ax=None, cor=True, diag=False, reg_family=None, hue_log=False,
                    show_cor_p=True, plot_range='each', return_ims=False, vmin=None, vmax=None,
                    cbar=True, cmap='jet', num_bin=20):
    # https://stackoverflow.com/questions/10439961/efficiently-create-a-density-plot-for-high-density-regions-points-for-sparse-re
    cor = _validate_boolean_flag(cor, "cor")
    diag = _validate_boolean_flag(diag, "diag")
    hue_log = _validate_boolean_flag(hue_log, "hue_log")
    show_cor_p = _validate_boolean_flag(show_cor_p, "show_cor_p")
    return_ims = _validate_boolean_flag(return_ims, "return_ims")
    cbar = _validate_boolean_flag(cbar, "cbar")
    if isinstance(num_bin, bool) or (not isinstance(num_bin, (int, np.integer))) or (num_bin <= 0):
        raise ValueError("num_bin must be a positive integer")
    for bound_name, bound_value in [('vmin', vmin), ('vmax', vmax)]:
        if bound_value is not None:
            if isinstance(bound_value, bool) or (not isinstance(bound_value, (int, float, np.integer, np.floating))):
                raise ValueError(f"{bound_name} must be None or a finite numeric value")
            if not np.isfinite(float(bound_value)):
                raise ValueError(f"{bound_name} must be None or a finite numeric value")
    if (vmin is not None) and (vmax is not None) and (float(vmin) > float(vmax)):
        raise ValueError("vmin must be less than or equal to vmax")
    if (df is not None) and (not hasattr(df, "columns")):
        raise ValueError("df must be a pandas DataFrame-like object with columns")
    if df is None:
        xlabel = 'x'
        ylabel = 'y'
        try:
            xval = np.asarray(x, dtype=float)
            yval = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("x and y must contain numeric values when df is None") from exc
        if xval.shape != yval.shape:
            raise ValueError("x and y must have the same shape when df is None")
        valid = np.isfinite(xval) & np.isfinite(yval)
        xval = xval[valid]
        yval = yval[valid]
        x_col = 'x'
        y_col = 'y'
        glm_df = None
    else:
        if not isinstance(x, str):
            raise ValueError("x must be a string column name when df is provided")
        if not isinstance(y, str):
            raise ValueError("y must be a string column name when df is provided")
        if (x not in df.columns) or (y not in df.columns):
            raise ValueError(f"df must include columns '{x}' and '{y}'")
        xlabel = x
        ylabel = y
        df_xy = df.loc[:, [x, y]].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        try:
            # Use positional columns so x==y works without ambiguous duplicate-name selection.
            x_numeric = pd.to_numeric(df_xy.iloc[:, 0], errors='raise').to_numpy(dtype=float)
            y_numeric = pd.to_numeric(df_xy.iloc[:, 1], errors='raise').to_numpy(dtype=float)
        except Exception as exc:
            raise ValueError(f"df columns '{x}' and '{y}' must contain numeric values") from exc
        finite_mask = np.isfinite(x_numeric) & np.isfinite(y_numeric)
        xval = x_numeric[finite_mask]
        yval = y_numeric[finite_mask]
        x_col = x
        y_col = y
        glm_df = pd.DataFrame({x_col: xval, y_col: yval})

    if xval.size == 0:
        raise ValueError("density_scatter received no finite data points")

    if reg_family is not None:
        if not hasattr(reg_family, "link"):
            raise ValueError("reg_family must be a statsmodels family object with a 'link' attribute")
        if glm_df is None:
            glm_df = pd.DataFrame({x_col: xval, y_col: yval})
        glm_formula = y_col + " ~ " + x_col
        try:
            mod = smf.glm(formula=glm_formula, data=glm_df, family=reg_family)
            res = mod.fit()
        except Exception as exc:
            raise ValueError(
                "GLM fit failed in density_scatter; check reg_family compatibility and input values"
            ) from exc
        xmin_predict = float(xval.min())
        xmax_predict = float(xval.max())
        if xmin_predict == xmax_predict:
            x_predict = np.array([xmin_predict], dtype=float)
        else:
            x_predict = np.linspace(xmin_predict, xmax_predict, num=100, endpoint=True)
        y_predict = res.predict({x_col: x_predict})
        link_name = reg_family.link.__class__.__name__.lower()
        if link_name == 'log':
            if np.any(y_predict <= 0):
                raise ValueError(
                    "GLM predicted non-positive values; cannot apply log transform in density_scatter"
                )
            y_predict = np.log(y_predict)
            with np.errstate(divide='ignore', invalid='ignore'):
                yval = np.log(yval)
            valid = np.isfinite(yval)
            xval = xval[valid]
            yval = yval[valid]
            if xval.size == 0:
                raise ValueError("density_scatter has no finite values after log transformation")
    bins = [num_bin, num_bin]  # number of bins
    thresh = 3  # density threshold
    if hue_log:
        thresh = np.log2(thresh)

    if not isinstance(plot_range, str):
        try:
            plot_range_len = len(plot_range)
        except TypeError as exc:
            raise ValueError("plot_range must be either 'each'/'ceil' or [xmin, xmax, ymin, ymax]") from exc
        if plot_range_len != 4:
            raise ValueError("plot_range must be either 'each'/'ceil' or [xmin, xmax, ymin, ymax]")
        try:
            xmin = float(plot_range[0])
            xmax = float(plot_range[1])
            ymin = float(plot_range[2])
            ymax = float(plot_range[3])
        except (TypeError, ValueError) as exc:
            raise ValueError("plot_range numeric bounds must be finite numbers") from exc
        if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (not np.isfinite(ymin)) or (not np.isfinite(ymax)):
            raise ValueError("plot_range numeric bounds must be finite numbers")
        if xmin > xmax:
            raise ValueError("plot_range requires xmin <= xmax")
        if ymin > ymax:
            raise ValueError("plot_range requires ymin <= ymax")
    else:
        if plot_range not in ('each', 'ceil'):
            raise ValueError("plot_range must be either 'each'/'ceil' or [xmin, xmax, ymin, ymax]")
        xmin = np.floor(xval.min())
        ymin = np.floor(yval.min())
        xmax = np.ceil(xval.max())
        ymax = np.ceil(yval.max())
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    if ymin == ymax:
        ymin -= 0.5
        ymax += 0.5

    xyrange = [[xmin, xmax], [ymin, ymax]]
    hh, locx, locy = np.histogram2d(xval, yval, range=xyrange, bins=bins)
    if hue_log:
        with np.errstate(divide='ignore'):
            hh = np.log2(hh)

    # Fast bin mapping avoids repeated searchsorted calls in np.digitize.
    xspan = xmax - xmin
    yspan = ymax - ymin
    if xspan == 0:
        x_idx = np.zeros(xval.shape[0], dtype=int)
    else:
        x_idx = np.floor((xval - xmin) * bins[0] / xspan).astype(int)
    if yspan == 0:
        y_idx = np.zeros(yval.shape[0], dtype=int)
    else:
        y_idx = np.floor((yval - ymin) * bins[1] / yspan).astype(int)
    x_idx = np.minimum(x_idx, bins[0] - 1)
    y_idx = np.minimum(y_idx, bins[1] - 1)
    ind = (x_idx >= 0) & (x_idx < bins[0]) & (y_idx >= 0) & (y_idx < bins[1])
    hhsub = hh[x_idx[ind], y_idx[ind]]  # values of the histogram where the points are
    xdat1 = xval[ind][hhsub < thresh]  # low density points
    ydat1 = yval[ind][hhsub < thresh]
    hh[hh < thresh] = np.nan  # fill the areas with low density by NaNs

    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True

    try:
        ims = ax.imshow(np.flipud(hh.T), cmap=cmap, extent=np.array(xyrange).flatten(),
                        vmin=vmin, vmax=vmax, interpolation='none', origin='upper', aspect="auto")
        ax.plot(xdat1, ydat1, '.', color='darkblue')

        if cbar:
            cbar = matplotlib.pyplot.colorbar(mappable=ims, ax=ax, format=matplotlib.ticker.ScalarFormatter(useMathText=True))
            cbar.ax.tick_params(axis='y', which='major', direction='out', length=3, width=1, pad=2)
            if hue_log:
                cbar.ax.text(0.5, 1.001, 'log$_2$ count', ha='center', va='bottom')
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.formatter.set_scientific(True)
            cbar.update_ticks()

        if reg_family is not None:
            ax.plot(x_predict, y_predict, '-', color="red", lw=2)

        if plot_range == 'ceil':
            xymin = min(xmin, ymin)
            xymax = max(xmax, ymax)
            ax.set_xlim(xymin, xymax)
            ax.set_ylim(xymin, xymax)
        else:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1, pad=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if cor:
            if show_cor_p:
                scor_val, spval_val = _spearmanr_fast(xval=xval, yval=yval)
                pcor_val, ppval_val = _pearsonr_fast(xval=xval, yval=yval)
                scor = f"{scor_val:.2f}"
                spval = '0' if spval_val == 0 else f"{spval_val:.2e}"
                pcor = f"{pcor_val:.2f}"
                ppval = '0' if ppval_val == 0 else f"{ppval_val:.2e}"
                title = "ρ=" + scor + " p=" + spval + ", r=" + pcor + " p=" + ppval
            else:
                scor_val, _ = _spearmanr_fast(xval=xval, yval=yval)
                pcor_val, _ = _pearsonr_fast(xval=xval, yval=yval)
                scor = f"{scor_val:.2f}"
                pcor = f"{pcor_val:.2f}"
                title = "ρ=" + scor + ", r=" + pcor
            ax.set_title(title, fontsize=matplotlib.rcParams['font.size'])

        if diag:
            diag_values = np.arange(min(xmin, ymin), max(xmax, ymax) + 1)
            ax.plot(diag_values, diag_values, '-', color='black', lw=1)
    except Exception:
        if created_internal_ax:
            matplotlib.pyplot.close(ax.figure)
        raise

    return ims if return_ims else ax


def hist_boxplot(x='', category='', df=None, colors=None, xlim=None, bins=None, alpha=0.9, box_step=0.15, ax=None):
    if df is None:
        df = pd.DataFrame()
    if not hasattr(df, "columns"):
        raise ValueError("df must be a pandas DataFrame-like object with columns")
    if (not isinstance(x, str)) or (x.strip() == ""):
        raise ValueError("x must be a non-empty string column name")
    if (not isinstance(category, str)) or (category.strip() == ""):
        raise ValueError("category must be a non-empty string column name")
    if isinstance(alpha, bool) or (not isinstance(alpha, (int, float, np.integer, np.floating))) or (not np.isfinite(float(alpha))):
        raise ValueError("alpha must be a finite numeric value between 0 and 1")
    alpha = float(alpha)
    if (alpha < 0) or (alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(box_step, bool) or (not isinstance(box_step, (int, float, np.integer, np.floating))) or (not np.isfinite(float(box_step))):
        raise ValueError("box_step must be a positive finite numeric value")
    box_step = float(box_step)
    if box_step <= 0:
        raise ValueError("box_step must be a positive finite numeric value")
    if colors is None:
        colors = {}
    if xlim is None:
        xlim = []
    if bins is None:
        bins = []
    if category not in df.columns:
        raise ValueError(f"category column '{category}' was not found in df")
    if x not in df.columns:
        raise ValueError(f"x column '{x}' was not found in df")
    _validate_hashable_column_values(data=df, column_name=category, argument_name='category')
    x_numeric = pd.to_numeric(df[x], errors='coerce')
    invalid_x_mask = df[x].notna() & x_numeric.isna()
    if invalid_x_mask.any():
        invalid_values = sorted(set(df.loc[invalid_x_mask, x].astype(str)))
        raise ValueError(f"x column '{x}' must contain numeric values; invalid values: {invalid_values}")
    non_finite_x_mask = x_numeric.notna() & (np.isinf(x_numeric))
    if non_finite_x_mask.any():
        invalid_values = sorted(set(df.loc[non_finite_x_mask, x].astype(str)))
        raise ValueError(f"x column '{x}' must contain finite numeric values; invalid values: {invalid_values}")
    x_numeric = x_numeric.astype(float)
    df = df.copy()
    df[x] = x_numeric
    try:
        xlim_len = len(xlim)
    except TypeError as exc:
        raise ValueError("xlim must be empty or contain exactly [xmin, xmax]") from exc
    if xlim_len not in (0, 2):
        raise ValueError("xlim must be empty or contain exactly [xmin, xmax]")
    if xlim_len == 2:
        try:
            xlim = [float(xlim[0]), float(xlim[1])]
        except (TypeError, ValueError) as exc:
            raise ValueError("xlim bounds must be finite numeric values") from exc
        if (not np.isfinite(xlim[0])) or (not np.isfinite(xlim[1])):
            raise ValueError("xlim bounds must be finite numeric values")
        if xlim[0] > xlim[1]:
            raise ValueError("xlim requires xmin <= xmax")
    category_values = df[category].dropna().drop_duplicates()
    if category_values.shape[0] == 0:
        raise ValueError(f"category column '{category}' must contain at least one non-NaN value")
    if isinstance(colors, dict) and (len(colors) > 0):
        observed_categories = df[category].dropna().tolist()
        missing_color_categories = [cv for cv in colors.keys() if cv not in observed_categories]
        if len(missing_color_categories) > 0:
            raise ValueError(
                f"colors contains categories that are not present in df[{category!r}]: {missing_color_categories}"
            )
    category_values = list(category_values)

    if len(xlim) == 0:
        x_values_all = df.loc[:, x].dropna()
        if x_values_all.empty:
            raise ValueError("hist_boxplot requires at least one non-NaN value in the x column")
        xlim = [float(x_values_all.min()), float(x_values_all.max())]
    if xlim[0] == xlim[1]:
        xlim = [xlim[0] - 0.5, xlim[1] + 0.5]
    if isinstance(bins, (int, np.integer)):
        if bins <= 0:
            raise ValueError("bins must be a positive integer when scalar")
        bins = np.linspace(xlim[0], xlim[1], num=(int(bins) + 1), endpoint=True)
    elif isinstance(bins, str):
        raise ValueError("bins must be empty, a positive integer, or an array-like of bin edges")
    else:
        try:
            bins_len = len(bins)
        except TypeError as exc:
            raise ValueError("bins must be empty, a positive integer, or an array-like of bin edges") from exc
        if bins_len == 0:
            bins = np.arange(xlim[0] - ((xlim[1] - xlim[0]) / 50), xlim[1] + ((xlim[1] - xlim[0]) / 50),
                                (xlim[1] - xlim[0]) / 100)
    try:
        bins = np.asarray(bins, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("bins must contain finite numeric bin-edge values") from exc
    if bins.shape[0] < 2:
        raise ValueError("bins must define at least 2 bin-edge values")
    if not np.isfinite(bins).all():
        raise ValueError("bins must contain finite numeric bin-edge values")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be strictly increasing")
    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True

    try:
        box_position = 1 + (box_step * len(category_values))
        yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        grouped_values = {
            key: group[x].dropna()
            for key, group in df.groupby(category, sort=False)
        }

        for i, cv in enumerate(category_values):
            label = cv
            if isinstance(colors, dict):
                color = colors.get(cv)
            elif isinstance(colors, np.ndarray):
                if colors.ndim == 0:
                    color = colors.item()
                else:
                    color = colors[i % len(colors)] if len(colors) > 0 else None
            elif isinstance(colors, (list, tuple)):
                color = colors[i % len(colors)] if len(colors) > 0 else None
            else:
                color = f'C{i}'
            if color is None:
                color = f'C{i}'
            x_values = grouped_values.get(cv, pd.Series(dtype=float))
            x_values_arr = x_values.to_numpy(copy=False)
            if x_values_arr.size == 0:
                raise ValueError(f"Category '{cv}' has no non-NaN values in column '{x}'")
            ax.hist(x_values_arr, bins=bins, cumulative=True, histtype='step', lw=1, alpha=alpha,
                    density=True, color=color, label=label)
            try:
                box = ax.boxplot(x_values_arr, positions=[box_position], orientation='horizontal', showfliers=False, widths=[0.1])
            except TypeError:
                box = ax.boxplot(x_values_arr, positions=[box_position], vert=False, showfliers=False, widths=[0.1])
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                matplotlib.pyplot.setp(box[element], color=color, linestyle='solid')
            yticks.append(box_position)
            box_position = box_position - box_step

        ax.set_xlabel(x)
        ax.set_ylabel('Cumulative frequency')
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


def ols_annotations(x, y, data=None, ax=None, color='black', font_size=8, textxy=[0.05, 0.95], textva='top',
                    textha='left', method='quantreg', stats=None):
    if stats is None:
        stats = ['N', 'slope', 'slope_p']
    elif isinstance(stats, str):
        stats = [stats]
    else:
        try:
            stats = list(stats)
        except TypeError as exc:
            raise ValueError("stats must be a string or a sequence of statistic names") from exc
    allowed_stats = {'N', 'slope', 'slope_p', 'rsquared', 'rsquared_p'}
    invalid_stats = [stat for stat in stats if (not isinstance(stat, str)) or (stat not in allowed_stats)]
    if len(invalid_stats) > 0:
        raise ValueError(f"stats contains unsupported entries: {invalid_stats}")
    try:
        textxy_len = len(textxy)
    except TypeError as exc:
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates") from exc
    if textxy_len != 2:
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates")
    try:
        textxy_x = float(textxy[0])
        textxy_y = float(textxy[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates") from exc
    if (not np.isfinite(textxy_x)) or (not np.isfinite(textxy_y)):
        raise ValueError("textxy must contain exactly [x, y] finite numeric coordinates")
    textxy = [textxy_x, textxy_y]
    if data is None:
        data = pd.DataFrame({'X': x, 'Y': y})
        x = 'X'
        y = 'Y'
    elif not hasattr(data, "columns"):
        raise ValueError("data must be a pandas DataFrame-like object with columns")
    if not isinstance(x, str):
        raise ValueError("x must be a string column name")
    if not isinstance(y, str):
        raise ValueError("y must be a string column name")
    if (x not in data.columns) or (y not in data.columns):
        raise ValueError(f"data must include columns '{x}' and '{y}'")
    if data.shape[0] < 2:
        raise ValueError("ols_annotations requires at least 2 rows")
    try:
        x_numeric = pd.to_numeric(data[x], errors='raise')
        y_numeric = pd.to_numeric(data[y], errors='raise')
    except Exception as exc:
        raise ValueError("ols_annotations requires numeric x and y values") from exc
    x_numeric = x_numeric.astype(float)
    y_numeric = y_numeric.astype(float)
    if (not np.isfinite(x_numeric.to_numpy()).all()) or (not np.isfinite(y_numeric.to_numpy()).all()):
        raise ValueError("ols_annotations requires finite numeric values in x and y")
    data = data.copy()
    data[x] = x_numeric
    data[y] = y_numeric
    data = data.sort_values(x)
    y_values = data[y].to_numpy(copy=False)
    y_has_variation = np.ptp(y_values) > 0

    res = None
    if method == 'ols':
        X = sm.add_constant(data.loc[:, x])
        Y = data.loc[:, y]
        mod = sm.OLS(Y, X)
        try:
            res = mod.fit()
        except Exception as exc:
            raise ValueError("ols fit failed in ols_annotations") from exc
    elif method == 'quantreg':
        mod = smf.quantreg(y + ' ~ ' + x, data)
        if y_has_variation:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
                        res = mod.fit(q=0.5)
            except Exception:
                res = None
    else:
        raise ValueError("method must be either 'ols' or 'quantreg'")

    N = data.shape[0]
    slope = np.nan
    slope_p = np.nan
    if (res is not None) and (('slope' in stats) or ('slope_p' in stats)):
        try:
            slope = res.params[x]
        except Exception:
            slope = np.nan
        try:
            slope_p = res.pvalues[x]
        except Exception:
            slope_p = np.nan
    rsquared = np.nan
    rsquared_p = np.nan
    if (res is not None) and (('rsquared' in stats) or ('rsquared_p' in stats)):
        if method == 'ols':
            df_resid = float(res.df_resid)
            if df_resid > 0 and y_has_variation:
                with np.errstate(divide='ignore', invalid='ignore'):
                    rsquared_candidate = res.rsquared_adj
                    rsquared_p_candidate = res.f_pvalue
                try:
                    rsquared = float(rsquared_candidate)
                except (TypeError, ValueError):
                    rsquared = np.nan
                if not np.isfinite(rsquared):
                    rsquared = np.nan
                try:
                    rsquared_p = float(rsquared_p_candidate)
                except (TypeError, ValueError):
                    rsquared_p = np.nan
                if not np.isfinite(rsquared_p):
                    rsquared_p = np.nan
        elif method == 'quantreg':
            with np.errstate(divide='ignore', invalid='ignore'):
                prsquared = getattr(res, 'prsquared', np.nan)
            try:
                rsquared = float(prsquared)
            except (TypeError, ValueError):
                rsquared = np.nan
            if not np.isfinite(rsquared):
                rsquared = np.nan

    text = ''
    for stat in stats:
        if stat == 'N':
            text += 'N = {:,}\n'.format(N)
        if stat == 'slope':
            text += 'slope = {}\n'.format('%.2f' % Decimal(slope))
        if stat == 'slope_p':
            text += 'P = {}\n'.format('%.2E' % Decimal(slope_p))
        if stat == 'rsquared':
            text += 'R2 = {}\n'.format('%.2f' % Decimal(rsquared))
        if stat == 'rsquared_p':
            text += 'P = {}\n'.format('%.2E' % Decimal(rsquared_p))
    created_internal_ax = False
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
        created_internal_ax = True
    try:
        ax.text(textxy[0], textxy[1], text, transform=ax.transAxes, va=textva, ha=textha, color=color, fontsize=font_size)
        x_endpoints = data[x].to_numpy(copy=False)[[0, N - 1]]
        if res is None:
            y_median = float(np.median(y_values))
            y_pred_endpoints = np.asarray([y_median, y_median], dtype=float)
        else:
            y_pred = np.asarray(res.predict(), dtype=float)
            y_pred_endpoints = y_pred[[0, N - 1]]
        ax.plot(x_endpoints, y_pred_endpoints, color=color)
    except Exception:
        if created_internal_ax:
            matplotlib.pyplot.close(ax.figure)
        raise
    return ax


if __name__ == "__main__":
    matplotlib.pyplot.interactive(False)
    nrow = 2
    ncol = 2
    fig, axes = matplotlib.pyplot.subplots(nrows=nrow, ncols=ncol, figsize=(4 * ncol, 4 * nrow))

    ax = axes[0, 0]
    x = np.random.normal(loc=0, scale=1, size=10000)
    y = x + np.random.normal(loc=1, scale=0.1, size=10000)
    density_scatter(x=x, y=y, ax=ax)
    fig.show()

    ax = axes[0, 1]
    x = np.random.normal(loc=0, scale=1, size=10000)
    y = x + np.random.normal(loc=1, scale=1, size=10000)
    density_scatter(x=x, y=y, ax=ax, diag=True)

    fig.tight_layout()
    fig.show()
