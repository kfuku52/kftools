import math
import numbers

import numpy as np
import scipy.stats as stats
from numpy.typing import ArrayLike, NDArray


def _validate_sample(sample: ArrayLike, sample_name: str) -> NDArray[np.float64]:
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


def _validate_bm_options(ttype, alpha):
    if isinstance(ttype, bool) or (not isinstance(ttype, numbers.Real)):
        raise ValueError("ttype must be a finite numeric value")
    ttype = float(ttype)
    if not math.isfinite(ttype):
        raise ValueError("ttype must be a finite numeric value")
    if isinstance(alpha, bool) or (not isinstance(alpha, numbers.Real)):
        raise ValueError("alpha must be a finite numeric value")
    alpha = float(alpha)
    if not math.isfinite(alpha):
        raise ValueError("alpha must be a finite numeric value")
    if (alpha <= 0) or (alpha >= 1):
        raise ValueError("alpha must be between 0 and 1")
    return ttype, alpha


def _bm_p_value(W, f_hat, ttype):
    if ttype < 0:
        return float(stats.t.cdf(W, f_hat))
    if ttype > 0:
        return float(stats.t.sf(W, f_hat))
    return float(2 * stats.t.sf(abs(W), f_hat))


def bm_test(
    x: ArrayLike,
    y: ArrayLike,
    ttype: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float, float, float, float]:
    """Run a Brunner-Munzel test and return its confidence interval.

    ``ttype=0`` selects a two-sided test, positive values test ``x < y``, and
    negative values test ``x > y``. The result is ``(statistic, dof, pvalue,
    probability_of_superiority, confidence_low, confidence_high)``.
    """

    # Copyright (C) 2016 Yukishige Shibata <y-shibat@mtd.biglobe.ne.jp>
    # All rights reserved.

    ttype, alpha = _validate_bm_options(ttype, alpha)
    x = _validate_sample(x, "x")
    y = _validate_sample(y, "y")
    N_x = x.size
    N_y = y.size

    cat_x_y = np.concatenate([x, y])

    R_total = stats.rankdata(cat_x_y, method="average")
    R_x = R_total[:N_x]
    R_y = R_total[N_x:]

    Ravr_x = float(np.mean(R_x))
    Ravr_y = float(np.mean(R_y))

    Pest = (Ravr_y - Ravr_x) / (N_x + N_y) + 0.5

    Ri_x = stats.rankdata(x, method="average")
    Ri_y = stats.rankdata(y, method="average")

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

    critical_t = float(stats.t.ppf(1 - (alpha / 2), f_hat))
    int_t = critical_t * math.sqrt((S2_x / (N_x * N_y * N_y)) + (S2_y / (N_x * N_x * N_y)))
    C_l = Pest - int_t
    C_h = Pest + int_t

    p_value = _bm_p_value(W, f_hat, ttype)
    return W, f_hat, p_value, Pest, C_l, C_h


def brunner_munzel_test(x: ArrayLike, y: ArrayLike, alternative: str = "two_sided") -> tuple[float, float]:
    """Return the Brunner-Munzel statistic and approximate t-distribution p-value.

    Missing and non-finite observations are discarded. ``alternative`` accepts
    ``"greater"``, ``"less"``, or ``"two_sided"`` and their documented aliases.
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
        raise ValueError(f"alternative must be one of {sorted(allowed_alternatives)}")
    ttype = {
        "greater": -1,
        "g": -1,
        "less": 1,
        "l": 1,
        "two_sided": 0,
    }[alternative_norm]
    statistic, _, probability, _, _, _ = bm_test(x, y, ttype=ttype)
    return statistic, probability
