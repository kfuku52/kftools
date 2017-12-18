import matplotlib.pyplot
import numpy
import pandas
import scipy.stats
import statsmodels.formula.api
from decimal import Decimal

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
