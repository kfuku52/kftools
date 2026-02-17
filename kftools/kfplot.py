import matplotlib.pyplot
import numpy
import pandas
import scipy.stats
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api
from decimal import Decimal


def _pearsonr_fast(xval, yval):
    xarr = numpy.asarray(xval, dtype=float)
    yarr = numpy.asarray(yval, dtype=float)
    if xarr.shape[0] != yarr.shape[0]:
        raise ValueError("xval and yval must have the same length")
    n = xarr.shape[0]
    if n < 2:
        return numpy.nan, numpy.nan

    x_center = xarr - xarr.mean()
    y_center = yarr - yarr.mean()
    x_norm = float(numpy.sqrt(numpy.dot(x_center, x_center)))
    y_norm = float(numpy.sqrt(numpy.dot(y_center, y_center)))
    if (x_norm == 0.0) or (y_norm == 0.0):
        return numpy.nan, numpy.nan

    r = float(numpy.dot(x_center, y_center) / (x_norm * y_norm))
    r = float(numpy.clip(r, -1.0, 1.0))
    if n < 3:
        return r, numpy.nan
    if abs(r) == 1.0:
        return r, 0.0
    t_stat = r * numpy.sqrt((n - 2) / (1 - (r * r)))
    pval = float(2 * scipy.stats.t.sf(abs(t_stat), n - 2))
    return r, pval


def _spearmanr_fast(xval, yval):
    rank_x = scipy.stats.rankdata(xval)
    rank_y = scipy.stats.rankdata(yval)
    return _pearsonr_fast(rank_x, rank_y)


def stacked_barplot(x, y, data, colors, ax):
    assert not all([isinstance(x, str), isinstance(y, str)])
    assert any([isinstance(x, list), isinstance(y, list)])
    assert any([isinstance(x, str), isinstance(y, str)])

    cols = {'x': x, 'y': y}
    dfs = {
        'x': pandas.DataFrame(data.loc[:, x]),
        'y': pandas.DataFrame(data.loc[:, y]),
    }
    for key, col in cols.items():
        if isinstance(col, list):
            dfs[key] = dfs[key].cumsum(axis=1)

    df = pandas.concat([dfs['x'], dfs['y']], axis=1)
    if isinstance(cols['x'], list):
        ncol = dfs['x'].columns.shape[0]
        for i in reversed(range(ncol)):
            seaborn.barplot(x=dfs['x'].columns[i], y=y, data=df, color=colors[i], ax=ax, linewidth=0)
    if isinstance(cols['y'], list):
        ncol = dfs['y'].columns.shape[0]
        for i in reversed(range(ncol)):
            seaborn.barplot(x=x, y=dfs['y'].columns[i], data=df, color=colors[i], ax=ax, linewidth=0)
    return ax


def density_scatter(x, y, df=None, ax=None, cor=True, diag=False, reg_family=None, hue_log=False,
                    show_cor_p=True, plot_range='each', return_ims=False, vmin=None, vmax=None,
                    cbar=True, cmap='jet', num_bin=20):
    # https://stackoverflow.com/questions/10439961/efficiently-create-a-density-plot-for-high-density-regions-points-for-sparse-re
    if df is None:
        xlabel = 'x'
        ylabel = 'y'
        xval = numpy.asarray(x, dtype=float)
        yval = numpy.asarray(y, dtype=float)
        valid = numpy.isfinite(xval) & numpy.isfinite(yval)
        xval = xval[valid]
        yval = yval[valid]
        x_col = 'x'
        y_col = 'y'
        glm_df = None
    else:
        xlabel = x
        ylabel = y
        df_xy = df.loc[:, [x, y]].replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis=0)
        xval = df_xy[x].astype(float).values
        yval = df_xy[y].astype(float).values
        x_col = x
        y_col = y
        glm_df = df_xy

    if xval.size == 0:
        raise ValueError("density_scatter received no finite data points")

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)

    if reg_family is not None:
        if glm_df is None:
            glm_df = pandas.DataFrame({x_col: xval, y_col: yval})
        glm_formula = y_col + " ~ " + x_col
        mod = statsmodels.formula.api.glm(formula=glm_formula, data=glm_df, family=reg_family)
        res = mod.fit()
        xmin_predict = float(xval.min())
        xmax_predict = float(xval.max())
        if xmin_predict == xmax_predict:
            x_predict = numpy.array([xmin_predict], dtype=float)
        else:
            x_predict = numpy.linspace(xmin_predict, xmax_predict, num=100, endpoint=True)
        y_predict = res.predict({x_col: x_predict})
        if 'log' in str(reg_family.link):
            y_predict = numpy.log(y_predict)
            print('log link function was detected in the GLM family. Y values are log-transformed.')
            yval = numpy.log(yval)
            valid = numpy.isfinite(yval)
            xval = xval[valid]
            yval = yval[valid]
    bins = [num_bin, num_bin]  # number of bins
    thresh = 3  # density threshold
    if hue_log:
        thresh = numpy.log2(thresh)

    if (not isinstance(plot_range, str)) and (len(plot_range) > 1):
        xmin = plot_range[0]
        xmax = plot_range[1]
        ymin = plot_range[2]
        ymax = plot_range[3]
    else:
        xmin = numpy.floor(xval.min())
        ymin = numpy.floor(yval.min())
        xmax = numpy.ceil(xval.max())
        ymax = numpy.ceil(yval.max())

    xyrange = [[xmin, xmax], [ymin, ymax]]
    hh, locx, locy = numpy.histogram2d(xval, yval, range=xyrange, bins=bins)
    if hue_log:
        with numpy.errstate(divide='ignore'):
            hh = numpy.log2(hh)

    # Fast bin mapping avoids repeated searchsorted calls in np.digitize.
    xspan = xmax - xmin
    yspan = ymax - ymin
    if xspan == 0:
        x_idx = numpy.zeros(xval.shape[0], dtype=int)
    else:
        x_idx = numpy.floor((xval - xmin) * bins[0] / xspan).astype(int)
    if yspan == 0:
        y_idx = numpy.zeros(yval.shape[0], dtype=int)
    else:
        y_idx = numpy.floor((yval - ymin) * bins[1] / yspan).astype(int)
    x_idx = numpy.minimum(x_idx, bins[0] - 1)
    y_idx = numpy.minimum(y_idx, bins[1] - 1)
    ind = (x_idx >= 0) & (x_idx < bins[0]) & (y_idx >= 0) & (y_idx < bins[1])
    hhsub = hh[x_idx[ind], y_idx[ind]]  # values of the histogram where the points are
    xdat1 = xval[ind][hhsub < thresh]  # low density points
    ydat1 = yval[ind][hhsub < thresh]
    hh[hh < thresh] = numpy.nan  # fill the areas with low density by NaNs

    ims = ax.imshow(numpy.flipud(hh.T), cmap=cmap, extent=numpy.array(xyrange).flatten(),
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
        print('vmin = ', ims.colorbar.vmin, ', vmax = ', ims.colorbar.vmax)

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
            rank_x = scipy.stats.rankdata(xval)
            rank_y = scipy.stats.rankdata(yval)
            scor = f"{numpy.corrcoef(rank_x, rank_y)[0, 1]:.2f}"
            pcor = f"{numpy.corrcoef(xval, yval)[0, 1]:.2f}"
            title = "ρ=" + scor + ", r=" + pcor
        ax.set_title(title, fontsize=matplotlib.rcParams['font.size'])

    if diag:
        diag_values = numpy.arange(min(xmin, ymin), max(xmax, ymax) + 1)
        ax.plot(diag_values, diag_values, '-', color='black', lw=1)

    return ims if return_ims else ax


def hist_boxplot(x='', category='', df=None, colors=None, xlim=None, bins=None, alpha=0.9, box_step=0.15, ax=None):
    if df is None:
        df = pandas.DataFrame()
    if colors is None:
        colors = {}
    if xlim is None:
        xlim = []
    if bins is None:
        bins = []
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
    category_values = df[category].drop_duplicates()
    if isinstance(colors, dict) and (len(colors) > 0):
        category_values = list(colors.keys())
    category_values = list(category_values)

    if len(xlim) == 0:
        x_values_all = df.loc[:, x].dropna()
        xlim = [float(x_values_all.min()), float(x_values_all.max())]
    if xlim[0] == xlim[1]:
        xlim = [xlim[0] - 0.5, xlim[1] + 0.5]
    if len(bins) == 0:
        bins = numpy.arange(xlim[0] - ((xlim[1] - xlim[0]) / 50), xlim[1] + ((xlim[1] - xlim[0]) / 50),
                            (xlim[1] - xlim[0]) / 100)

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
        elif isinstance(colors, list):
            color = colors[i % len(colors)] if len(colors) > 0 else None
        else:
            color = f'C{i}'
        if color is None:
            color = f'C{i}'
        x_values = grouped_values.get(cv, pandas.Series(dtype=float))
        x_values_arr = x_values.to_numpy(copy=False)
        ax.hist(x_values_arr, bins=bins, cumulative=True, histtype='step', lw=1, alpha=alpha,
                density=True, color=color, label=label)
        box = ax.boxplot(x_values_arr, positions=[box_position], vert=False, showfliers=False, widths=[0.1])
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            matplotlib.pyplot.setp(box[element], color=color, linestyle='solid')
        yticks.append(box_position)
        box_position = box_position - box_step

    ax.set_xlabel(x)
    ax.set_ylabel('Cumulative frequency')
    ax.set_xlim(numpy.mean([xlim[0], min(bins)]), numpy.mean([xlim[1], max(bins)]))
    ax.set_ylim(-0.02, 1.1 + (box_step * len(category_values)))
    ax.set_yticks(yticks)
    yticklabels = [y for y in yticks if y <= 1] + category_values
    ax.set_yticklabels(yticklabels)
    return ax


def ols_annotations(x, y, data=None, ax=None, color='black', font_size=8, textxy=[0.05, 0.95], textva='top',
                    textha='left', method='quantreg', stats=None):
    if stats is None:
        stats = ['N', 'slope', 'slope_p']
    if ax is None:
        _, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5, 5), sharex=False)
    if data is None:
        data = pandas.DataFrame({'X': x, 'Y': y})
        x = 'X'
        y = 'Y'
    data = data.sort_values(x)

    if method == 'ols':
        X = sm.add_constant(data.loc[:, x])
        Y = data.loc[:, y]
        mod = sm.OLS(Y, X)
        res = mod.fit()
    elif method == 'quantreg':
        mod = statsmodels.formula.api.quantreg(y + ' ~ ' + x, data)
        res = mod.fit(q=0.5)

    N = data.shape[0]
    slope = res.params[x]
    slope_p = res.pvalues[x]
    rsquared = res.rsquared_adj
    rsquared_p = res.f_pvalue

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
    ax.text(textxy[0], textxy[1], text, transform=ax.transAxes, va=textva, ha=textha, color=color, fontsize=font_size)
    ax.plot(data[x].values[[0, N - 1]], res.predict()[[0, N - 1]], color=color)


if __name__ == "__main__":
    matplotlib.pyplot.interactive(False)
    nrow = 2
    ncol = 2
    fig, axes = matplotlib.pyplot.subplots(nrows=nrow, ncols=ncol, figsize=(4 * ncol, 4 * nrow))

    ax = axes[0, 0]
    x = numpy.random.normal(loc=0, scale=1, size=10000)
    y = x + numpy.random.normal(loc=1, scale=0.1, size=10000)
    density_scatter(x=x, y=y, ax=ax)
    fig.show()

    ax = axes[0, 1]
    x = numpy.random.normal(loc=0, scale=1, size=10000)
    y = x + numpy.random.normal(loc=1, scale=1, size=10000)
    density_scatter(x=x, y=y, ax=ax, diag=True)

    fig.tight_layout()
    fig.show()
