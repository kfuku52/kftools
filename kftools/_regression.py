"""Numerical results for plotting annotations, independent of Matplotlib."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AnnotationFit:
    predictions: NDArray[np.float64]
    statistics: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(
            ("slope", "slope_p", "rsquared", "rsquared_adj", "rsquared_p"), float("nan")
        )
    )


def _finite_scalar(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _fit_statistics(result, method: str, scale: float, y_has_variation: bool) -> dict[str, float]:
    values = {
        "slope": _finite_scalar(result.params[1] / scale),
        "slope_p": float("nan"),
        "rsquared": float("nan"),
        "rsquared_adj": float("nan"),
        "rsquared_p": float("nan"),
    }
    # An exact two-point fit has a descriptive R2 but no residual degrees of
    # freedom for coefficient inference or adjusted R2.
    with np.errstate(divide="ignore", invalid="ignore"):
        if result.df_resid > 0:
            values["slope_p"] = _finite_scalar(result.pvalues[1])
        if y_has_variation:
            if method == "ols":
                values["rsquared"] = _finite_scalar(result.rsquared)
                if result.df_resid > 0:
                    values["rsquared_adj"] = _finite_scalar(result.rsquared_adj)
                    values["rsquared_p"] = _finite_scalar(result.f_pvalue)
            else:
                values["rsquared"] = _finite_scalar(result.prsquared)
    return values


def fit_annotation_model(x: NDArray[np.float64], y: NDArray[np.float64], method: str) -> AnnotationFit:
    """Fit OLS or median regression, explicitly handling unidentified slopes."""
    if method not in {"ols", "quantreg"}:
        raise ValueError("method must be either 'ols' or 'quantreg'")
    y_has_variation = bool(np.ptp(y) > 0)
    if np.ptp(x) == 0:
        intercept = float(np.mean(y) if method == "ols" else np.median(y))
        return AnnotationFit(np.full(y.shape, intercept))
    if not y_has_variation:
        fit = AnnotationFit(np.full(y.shape, y[0]))
        fit.statistics["slope"] = 0.0
        return fit

    # Internal array columns avoid collisions with user column names ("const"
    # included). Centering/scaling also avoids rank loss for large x offsets.
    centered = x - x.mean()
    scale = float(np.max(np.abs(centered)))
    predictor = centered / scale
    exog = np.column_stack((np.ones(x.size), predictor))

    import statsmodels.api as sm

    try:
        if method == "ols":
            result = sm.OLS(y, exog).fit()
        else:
            # QuantReg's covariance estimate can be undefined with a zero
            # residual bandwidth. Keep the fitted line and report NaN inference;
            # model/convergence warnings are deliberately not suppressed.
            with np.errstate(divide="ignore", invalid="ignore"):
                result = sm.QuantReg(y, exog).fit(q=0.5)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{method} fit failed in ols_annotations") from exc
    return AnnotationFit(
        np.asarray(result.predict(), dtype=float), _fit_statistics(result, method, scale, y_has_variation)
    )
