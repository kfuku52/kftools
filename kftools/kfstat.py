import scipy.stats
import numpy

def brunner_munzel_test(x, y, alternative="two_sided"):
    # http://codegists.com/snippet/python/brunner_munzel_testpy_katsuyaito_python
    """
    Computes the Brunner Munzel statistic

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    x : sequence
        Inumpyut
    y : sequence
        Inumpyut
    alternative : {greater, less, two_sided }

    Returns
    -------
    statistic : float
        The Brunner Munzel  statistics
    pvalue : float
        Approximate p-value assuming a t distribution.

     """
    x = numpy.ma.asarray(x).compressed().view(numpy.ndarray)
    y = numpy.ma.asarray(y).compressed().view(numpy.ndarray)
    ranks = scipy.stats.rankdata(numpy.concatenate([x, y]))
    (nx, ny) = (len(x), len(y))
    rankx = scipy.stats.rankdata(x)
    ranky = scipy.stats.rankdata(y)
    rank_mean1 = numpy.mean(ranks[0:nx])
    rank_mean2 = numpy.mean(ranks[nx:nx + ny])
    pst = (rank_mean2 - (ny + 1) / 2) / nx
    v1_set = [(i - j - rank_mean1 + (nx + 1) / 2) ** 2 for (i, j) in zip(ranks[0:nx], rankx)]
    v2_set = [(i - j - rank_mean2 + (ny + 1) / 2) ** 2 for (i, j) in zip(ranks[nx:nx + ny], ranky)]
    v1 = numpy.sum(v1_set) / (nx - 1)
    v2 = numpy.sum(v2_set) / (ny - 1)
    statistic = nx * ny * (rank_mean2 - rank_mean1) / (nx + ny) / numpy.sqrt(nx * v1 + ny * v2)
    dfbm = ((nx * v1 + ny * v2) ** 2) / (((nx * v1) ** 2) / (nx - 1) + ((ny * v2) ** 2) / (ny - 1))
    if ((alternative == "greater") | (alternative == "g")):
        prob = scipy.stats.t.cdf(statistic, dfbm)
    elif ((alternative == "less") | (alternative == "l")):
        prob = 1 - scipy.stats.t.cdf(statistic, dfbm)
    else:
        alternative = "two_sided"
        abst = numpy.abs(statistic)
        prob = scipy.stats.t.cdf(abst, dfbm)
        prob = 2 * min(prob, 1 - prob)
    return statistic, prob

if __name__=="__main__":
    x = numpy.random.normal(loc=0, scale=1, size=10000)
    y = x + numpy.random.normal(loc=1, scale=1, size=10000)
    out = brunner_munzel_test(x=x, y=y, alternative="two_sided")
    print(out)