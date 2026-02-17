import math

import numpy
import scipy.stats


def _calc_S2(R, Ri, Ravr):
    # Copyright (C) 2016 Yukishige Shibata <y-shibat@mtd.biglobe.ne.jp>
    # All rights reserved.
    R = numpy.asarray(R, dtype=float)
    Ri = numpy.asarray(Ri, dtype=float)
    if len(Ri) != len(R):
        raise ValueError("number of elements in sample and its rank must be same.")

    num_elem = Ri.size
    v = R - Ri - Ravr + (num_elem + 1.0) / 2.0
    return float(numpy.dot(v, v) / (num_elem - 1))


def bm_test(x, y, ttype=0, alpha=0.05):
    # Copyright (C) 2016 Yukishige Shibata <y-shibat@mtd.biglobe.ne.jp>
    # All rights reserved.

    """This function implements the Brunner-Munzel test.
    Arguments 'x' and 'y' are list/scipy array whose type is integer or floaing point.
    Run Brunner-Munzel test aginst 2 list of samples, X and Y.
    All of element in both X and Y must be valid number.
    You can select phypothesis by specifying negative, 0, or positive integer to 'ttype'.
    0 is for null hypothesis, -1 or smaller integer is for less hypotesis, 1 or bigger for greater one.
    Default is 0.
    'alpha' is the confidential level. Devault is 0.05. Its for 95% confidential interval.
    This function returns a tuple of (W, dof, p, Pest), where
       W: test statisfic
       dof: degree of freedom
       p: p-value
       Pest: numerial list of the range of given condifidential level.
    The implementation is based on the description in next URL.
      http://oku.edu.mie-u.ac.jp/~okumura/stat/brunner-munzel.html
    """

    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    N_x = x.size
    N_y = y.size

    cat_x_y = numpy.concatenate([x, y])

    R_total = scipy.stats.rankdata(cat_x_y, method='average')
    R_x = R_total[:N_x]
    R_y = R_total[N_x:]

    Ravr_x = float(numpy.mean(R_x))
    Ravr_y = float(numpy.mean(R_y))

    Pest = (Ravr_y - Ravr_x) / (N_x + N_y) + 0.5

    Ri_x = scipy.stats.rankdata(x, method='average')
    Ri_y = scipy.stats.rankdata(y, method='average')

    S2_x = _calc_S2(R_x, Ri_x, Ravr_x)
    S2_y = _calc_S2(R_y, Ri_y, Ravr_y)

    W = ((N_x * N_y) * (Ravr_y - Ravr_x)) / ((N_x + N_y) * math.sqrt(N_x * S2_x + N_y * S2_y))

    nS2_x = N_x * S2_x
    nS2_y = N_y * S2_y

    f_hat_num = (nS2_x + nS2_y) * (nS2_x + nS2_y)
    f_hat_den = (nS2_x * nS2_x) / (N_x - 1) + (nS2_y * nS2_y) / (N_y - 1)
    f_hat = f_hat_num / f_hat_den

    int_t = scipy.stats.t.ppf(1 - (alpha / 2), f_hat) * math.sqrt(
        (S2_x / (N_x * N_y * N_y)) + (S2_y / (N_x * N_x * N_y))
    )
    C_l = Pest - int_t
    C_h = Pest + int_t

    if ttype < 0:
        p_value = scipy.stats.t.cdf(W, f_hat)
    elif ttype > 0:
        p_value = 1 - scipy.stats.t.cdf(W, f_hat)
    else:
        pt = scipy.stats.t.cdf(abs(W), f_hat)
        p_value = 2 * min(pt, 1 - pt)

    return W, f_hat, p_value, Pest, C_l, C_h


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
    nx, ny = len(x), len(y)
    rankx = scipy.stats.rankdata(x)
    ranky = scipy.stats.rankdata(y)
    rank_mean1 = numpy.mean(ranks[0:nx])
    rank_mean2 = numpy.mean(ranks[nx:nx + ny])
    v1 = numpy.sum((ranks[0:nx] - rankx - rank_mean1 + (nx + 1) / 2) ** 2) / (nx - 1)
    v2 = numpy.sum((ranks[nx:nx + ny] - ranky - rank_mean2 + (ny + 1) / 2) ** 2) / (ny - 1)
    statistic = nx * ny * (rank_mean2 - rank_mean1) / (nx + ny) / numpy.sqrt(nx * v1 + ny * v2)
    dfbm = ((nx * v1 + ny * v2) ** 2) / (((nx * v1) ** 2) / (nx - 1) + ((ny * v2) ** 2) / (ny - 1))
    if alternative in ("greater", "g"):
        prob = scipy.stats.t.cdf(statistic, dfbm)
    elif alternative in ("less", "l"):
        prob = 1 - scipy.stats.t.cdf(statistic, dfbm)
    else:
        abst = numpy.abs(statistic)
        prob = scipy.stats.t.cdf(abst, dfbm)
        prob = 2 * min(prob, 1 - prob)
    return statistic, prob


if __name__ == "__main__":
    x = numpy.random.normal(loc=0, scale=1, size=10000)
    y = x + numpy.random.normal(loc=1, scale=1, size=10000)
    out = brunner_munzel_test(x=x, y=y, alternative="two_sided")
    print(out)
