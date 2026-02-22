import math
import numbers

import numpy as np
import scipy.stats as stats


def _validate_sample(sample, sample_name):
    try:
        arr = np.asarray(sample, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_name} must contain numeric values") from exc
    if arr.size < 2:
        raise ValueError(f"{sample_name} must contain at least 2 values")
    if not np.isfinite(arr).all():
        raise ValueError(f"{sample_name} must contain only finite numeric values")
    return arr


def _calc_S2(R, Ri, Ravr):
    # Copyright (C) 2016 Yukishige Shibata <y-shibat@mtd.biglobe.ne.jp>
    # All rights reserved.
    R = np.asarray(R, dtype=float)
    Ri = np.asarray(Ri, dtype=float)
    if len(Ri) != len(R):
        raise ValueError("number of elements in sample and its rank must be same.")

    num_elem = Ri.size
    v = R - Ri - Ravr + (num_elem + 1.0) / 2.0
    return float(np.dot(v, v) / (num_elem - 1))


def bm_test(x, y, ttype=0, alpha=0.05):
    # Copyright (C) 2016 Yukishige Shibata <y-shibat@mtd.biglobe.ne.jp>
    # All rights reserved.

    """This function implements the Brunner-Munzel test.
    Arguments 'x' and 'y' are list/scipy array whose type is integer or floaing point.
    Run Brunner-Munzel test aginst 2 list of samples, X and Y.
    All of element in both X and Y must be valid number.
    You can select phypothesis by specifying negative, 0, or positive integer to 'ttype'.
    0 is for two-sided null hypothesis.
    1 or larger corresponds to the one-sided "less" alternative (x < y),
    and -1 or smaller corresponds to the one-sided "greater" alternative (x > y).
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

    if isinstance(ttype, bool) or (not isinstance(ttype, numbers.Real)) or (not np.isfinite(ttype)):
        raise ValueError("ttype must be a finite numeric value")
    if isinstance(alpha, bool) or (not isinstance(alpha, numbers.Real)) or (not np.isfinite(alpha)):
        raise ValueError("alpha must be a finite numeric value")
    if (alpha <= 0) or (alpha >= 1):
        raise ValueError("alpha must be between 0 and 1")
    x = _validate_sample(x, "x")
    y = _validate_sample(y, "y")
    N_x = x.size
    N_y = y.size

    cat_x_y = np.concatenate([x, y])

    R_total = stats.rankdata(cat_x_y, method='average')
    R_x = R_total[:N_x]
    R_y = R_total[N_x:]

    Ravr_x = float(np.mean(R_x))
    Ravr_y = float(np.mean(R_y))

    Pest = (Ravr_y - Ravr_x) / (N_x + N_y) + 0.5

    Ri_x = stats.rankdata(x, method='average')
    Ri_y = stats.rankdata(y, method='average')

    S2_x = _calc_S2(R_x, Ri_x, Ravr_x)
    S2_y = _calc_S2(R_y, Ri_y, Ravr_y)

    variance_term = N_x * S2_x + N_y * S2_y
    if variance_term <= 0:
        raise ValueError("Brunner-Munzel test is undefined when pooled variance is zero")
    w_denominator = (N_x + N_y) * math.sqrt(variance_term)
    if w_denominator == 0:
        raise ValueError("Brunner-Munzel test is undefined because the test denominator is zero")
    W = ((N_x * N_y) * (Ravr_y - Ravr_x)) / w_denominator

    nS2_x = N_x * S2_x
    nS2_y = N_y * S2_y

    f_hat_num = (nS2_x + nS2_y) * (nS2_x + nS2_y)
    f_hat_den = (nS2_x * nS2_x) / (N_x - 1) + (nS2_y * nS2_y) / (N_y - 1)
    if f_hat_den == 0:
        raise ValueError("Brunner-Munzel test is undefined because the degree-of-freedom denominator is zero")
    f_hat = f_hat_num / f_hat_den

    int_t = stats.t.ppf(1 - (alpha / 2), f_hat) * math.sqrt(
        (S2_x / (N_x * N_y * N_y)) + (S2_y / (N_x * N_x * N_y))
    )
    C_l = Pest - int_t
    C_h = Pest + int_t

    if ttype < 0:
        p_value = stats.t.cdf(W, f_hat)
    elif ttype > 0:
        p_value = stats.t.sf(W, f_hat)
    else:
        p_value = 2 * stats.t.sf(abs(W), f_hat)

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
    try:
        x = np.asarray(np.ma.asarray(x).compressed().view(np.ndarray), dtype=float).reshape(-1)
        y = np.asarray(np.ma.asarray(y).compressed().view(np.ndarray), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("x and y must contain numeric values") from exc
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2:
        raise ValueError("x must contain at least 2 values after removing missing/non-finite entries")
    if y.size < 2:
        raise ValueError("y must contain at least 2 values after removing missing/non-finite entries")
    if not isinstance(alternative, str):
        raise ValueError("alternative must be a string")
    alternative_norm = alternative.lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    allowed_alternatives = {"greater", "g", "less", "l", "two_sided"}
    if alternative_norm not in allowed_alternatives:
        raise ValueError(
            f"alternative must be one of {sorted(allowed_alternatives)}"
        )
    ranks = stats.rankdata(np.concatenate([x, y]))
    nx, ny = len(x), len(y)
    rankx = stats.rankdata(x)
    ranky = stats.rankdata(y)
    rank_mean1 = np.mean(ranks[0:nx])
    rank_mean2 = np.mean(ranks[nx:nx + ny])
    v1 = np.sum((ranks[0:nx] - rankx - rank_mean1 + (nx + 1) / 2) ** 2) / (nx - 1)
    v2 = np.sum((ranks[nx:nx + ny] - ranky - rank_mean2 + (ny + 1) / 2) ** 2) / (ny - 1)
    variance_term = (nx * v1) + (ny * v2)
    if variance_term <= 0:
        raise ValueError("Brunner-Munzel test is undefined when pooled variance is zero")
    statistic = nx * ny * (rank_mean2 - rank_mean1) / (nx + ny) / np.sqrt(variance_term)
    dfbm_den = (((nx * v1) ** 2) / (nx - 1) + ((ny * v2) ** 2) / (ny - 1))
    if dfbm_den == 0:
        raise ValueError("Brunner-Munzel test is undefined because the degree-of-freedom denominator is zero")
    dfbm = ((nx * v1 + ny * v2) ** 2) / dfbm_den
    if alternative_norm in ("greater", "g"):
        prob = stats.t.cdf(statistic, dfbm)
    elif alternative_norm in ("less", "l"):
        prob = stats.t.sf(statistic, dfbm)
    else:
        prob = 2 * stats.t.sf(np.abs(statistic), dfbm)
    return statistic, prob


if __name__ == "__main__":
    x = np.random.normal(loc=0, scale=1, size=10000)
    y = x + np.random.normal(loc=1, scale=1, size=10000)
    out = brunner_munzel_test(x=x, y=y, alternative="two_sided")
    print(out)
