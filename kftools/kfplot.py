import matplotlib.pyplot
import seaborn
import numpy
import pandas
import scipy.stats
import statsmodels.formula.api
from decimal import Decimal

def stacked_barplot(x, y, data, colors, ax):
    assert not all([isinstance(x, str), isinstance(y, str)])
    assert any([isinstance(x, list), isinstance(y, list)])
    assert any([isinstance(x, str), isinstance(y, str)])
    cols = {'x':x, 'y':y}
    dfs = dict()
    dfs['x'] = pandas.DataFrame(data.loc[:,x])
    dfs['y'] = pandas.DataFrame(data.loc[:,y])
    for k in cols.keys():
        if isinstance(cols[k], list):
            ncol = dfs[k].columns.shape[0]
            for i in reversed(range(ncol)):
                dfs[k].iloc[:,i] = dfs[k].iloc[:,:i+1].sum(axis=1)
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
                    cbar=True):
    # https://stackoverflow.com/questions/10439961/efficiently-create-a-density-plot-for-high-density-regions-points-for-sparse-re
    if df is None:
        df = pandas.DataFrame()
        df['x'] = x
        df['y'] = y
        x = 'x'
        y = 'y'
    df = df.loc[:,[x,y]].replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis=0)
    if ax is None:
        fig,ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(5,5), sharex=False)
    if not reg_family is None:
        glm_formula = y+" ~ "+x
        mod = statsmodels.formula.api.glm(formula=glm_formula, data=df, family=reg_family)
        res = mod.fit()
        x_predict = numpy.arange(df[x].min(),df[x].max(),(df[x].max()-df[x].min())/100)
        y_predict = res.predict({x:x_predict})
        if 'log' in str(reg_family.link):
            y_predict = numpy.log(y_predict)
            print('log link function was detected in the GLM family. Y values are log-transformed.')
            df.loc[:,y] = numpy.log(df[y])
            df = df.loc[:,[x,y]].replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis=0)
    xval=df[x].astype(float)
    yval=df[y].astype(float)
    bins = [100,100] # number of bins
    thresh = 3  #density threshold
    if hue_log:
        thresh = numpy.log2(thresh)
    if (len(plot_range)>1)&(type(plot_range)!=str):
        xmin = plot_range[0]
        xmax = plot_range[1]
        ymin = plot_range[2]
        ymax = plot_range[3]
    else:
        xmin=numpy.floor(xval.min())
        ymin=numpy.floor(yval.min())
        xmax=numpy.ceil(xval.max())
        ymax=numpy.ceil(yval.max())
    xyrange=[[xmin,xmax],[ymin,ymax]]
    hh, locx, locy = scipy.histogram2d(xval, yval, range=xyrange, bins=bins)
    if hue_log:
        hh = numpy.log2(hh)
    posx = numpy.digitize(xval, locx)
    posy = numpy.digitize(yval, locy)
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xval[ind][hhsub < thresh] # low density points
    ydat1 = yval[ind][hhsub < thresh]
    hh[hh < thresh] = numpy.nan # fill the areas with low density by NaNs
    ims = ax.imshow(numpy.flipud(hh.T),cmap='jet',extent=numpy.array(xyrange).flatten(), vmin=vmin, vmax=vmax, interpolation='none', origin='upper', aspect="auto")
    ax.plot(xdat1, ydat1, '.',color='darkblue')
    if cbar:
        cbar = matplotlib.pyplot.colorbar(mappable=ims, ax=ax, format=matplotlib.ticker.ScalarFormatter(useMathText=True))
        cbar.ax.tick_params(axis='y', which='major', direction='out', length=3, width=1, pad=2)
        if hue_log:
            cbar.ax.text(0.5,1.001, 'log$_2$ count', ha='center', va='bottom')
        cbar.formatter.set_powerlimits((-2, 2))
        cbar.formatter.set_scientific(True)
        cbar.update_ticks()
        print('vmin = ', ims.colorbar.vmin, ', vmax = ', ims.colorbar.vmax)
    if not reg_family is None:
        ax.plot(x_predict, y_predict, '-', color="red", lw=2)
    if plot_range=='ceil':
        xymin = min(xmin, ymin)
        xymax = max(xmax, ymax)
        ax.set_xlim(xymin, xymax)
        ax.set_ylim(xymin, xymax)
    else:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1, pad=2)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if cor:
        s_out = scipy.stats.spearmanr(a=xval, b=yval)
        p_out = scipy.stats.pearsonr(x=xval, y=yval)
        scor = str(Decimal(str(numpy.round(s_out.correlation, decimals=2))).quantize(Decimal('1.00')))
        spval = "{:.2e}".format(Decimal(str(s_out.pvalue)))
        if spval=='0.00e+1':
            spval = '0'
        pcor = str(Decimal(str(numpy.round(p_out[0], decimals=2))).quantize(Decimal('1.00')))
        ppval = "{:.2e}".format(Decimal(str(p_out[1])))
        if ppval=='0.00e+1':
            ppval = '0'
        if show_cor_p:
            title="ρ="+scor+" p="+spval+", r="+pcor+" p="+ppval
        else:
            title="ρ="+scor+", r="+pcor
        ax.set_title(title, fontsize=matplotlib.rcParams['font.size'])
    if diag:
        diag_values = numpy.arange(min(xmin,ymin), max(xmax,ymax)+1)
        ax.plot(diag_values, diag_values, "r-", color='black', lw=1)
    if return_ims:
        return ims
    else:
        return ax

def hist_boxplot(x='', category='', df=pandas.DataFrame(), colors={}, xlim=[], bins=[], alpha=0.9, box_step=0.15, ax=None):
    category_values = df[category].drop_duplicates()
    if isinstance(colors, dict):
        category_values = list(colors.keys())
    box_position = 1 + (box_step*len(category_values))
    yticks = [0.0,0.2,0.4,0.6,0.8,1.0]
    x_values = dict()
    x_nums = dict()
    bins=numpy.arange(xlim[0]-((xlim[1]-xlim[0])/50), xlim[1]+((xlim[1]-xlim[0])/50), (xlim[1]-xlim[0])/100)
    for cv in category_values:
        label = cv
        if isinstance(colors, dict):
            color = colors[cv]
        elif isinstance(colors, list):
            color = colors.pop()
        df_tmp=df.loc[(df[category]==cv),:]
        x_values[cv] = df_tmp[x].dropna()
        x_nums[cv] = df_tmp[x].dropna().shape[0]
        hist_kws={'cumulative':True,'histtype':'step','lw':1,'alpha':alpha}
        seaborn.distplot(x_values[cv], color=color, kde=False, bins=bins, ax=ax, hist_kws=hist_kws, norm_hist=True, label=label)
        box = ax.boxplot(x_values[cv].tolist(), positions=[box_position,], vert=False, showfliers=False, widths=[0.1,])
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            matplotlib.pyplot.setp(box[element], color=color, linestyle='solid')
        yticks.append(box_position)
        box_position = box_position - box_step
    ax.set_xlabel(x)
    ax.set_ylabel('Cumulative frequency')
    ax.set_xlim(numpy.mean([xlim[0],min(bins)]),numpy.mean([xlim[1],max(bins)]))
    ax.set_ylim(-0.02, 1.1+(box_step*len(category_values)))
    ax.set_yticks(yticks)
    yticklabels = [ y for y in yticks if y<=1 ] + category_values
    ax.set_yticklabels(yticklabels)
    return ax


if __name__=="__main__":
    matplotlib.pyplot.interactive(False)
    nrow = 2
    ncol = 2
    fig, axes = matplotlib.pyplot.subplots(nrows=nrow, ncols=ncol, figsize=(4*ncol, 4*nrow))

    ax = axes[0,0]
    x = numpy.random.normal(loc=0, scale=1, size=10000)
    y = x + numpy.random.normal(loc=1, scale=0.1, size=10000)
    density_scatter(x=x, y=y, ax=ax)
    fig.show()

    ax = axes[0,1]
    x = numpy.random.normal(loc=0, scale=1, size=10000)
    y = x + numpy.random.normal(loc=1, scale=1, size=10000)
    density_scatter(x=x, y=y, ax=ax, diag=True)

    fig.tight_layout()
    fig.show()
