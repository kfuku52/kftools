from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def calc_complementarity(array1: ArrayLike, array2: ArrayLike) -> float:
    """Return the mean relative difference between two non-negative profiles.

    Inputs are flattened and must contain equal, non-zero numbers of finite
    values. Each pair contributes ``abs(a - b) / max(a, b)``; a pair of zeros
    contributes zero. Inputs are not modified.
    """
    try:
        arr1 = np.asarray(array1, dtype=float).reshape(-1)
        arr2 = np.asarray(array2, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("array1 and array2 must contain numeric values") from exc
    if (not np.isfinite(arr1).all()) or (not np.isfinite(arr2).all()):
        raise ValueError("array1 and array2 must contain only finite numeric values")
    if arr1.size == 0 or arr2.size == 0:
        raise ValueError("array1 and array2 must each contain at least one value")
    if arr1.size != arr2.size:
        raise ValueError("array1 and array2 must contain the same number of values")
    if np.any(arr1 < 0) or np.any(arr2 < 0):
        raise ValueError("array1 and array2 must contain non-negative values")
    max_values = np.maximum(arr1, arr2)
    abs_diff = np.abs(arr1 - arr2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.divide(abs_diff, max_values, out=np.zeros_like(abs_diff), where=(max_values != 0))
    normalized_dif = rel_diff.mean()
    return float(normalized_dif)


def _validate_tau_columns(df, columns):
    if not hasattr(df, "columns"):
        raise ValueError("df must be a pandas DataFrame-like object with columns")
    if columns is None:
        raise ValueError("columns must contain at least one column name")
    if isinstance(columns, str):
        columns = [columns]
    else:
        try:
            columns = list(columns)
        except TypeError as exc:
            raise ValueError("columns must be a non-empty sequence of column names") from exc
    if len(columns) == 0:
        raise ValueError("columns must contain at least one column name")
    invalid_column_names = [col for col in columns if (not isinstance(col, str)) or (col.strip() == "")]
    if len(invalid_column_names) > 0:
        raise ValueError(f"columns must contain non-empty string column names; invalid entries: {invalid_column_names}")
    if len(set(columns)) != len(columns):
        raise ValueError("columns must not contain duplicate column names")
    missing_columns = [col for col in columns if col not in df.columns]
    if len(missing_columns) > 0:
        raise ValueError(f"columns not found in dataframe: {missing_columns}")
    return columns


def _prepare_tau_matrix(df, columns, unlog2, unPlus1):
    try:
        x = df.loc[:, columns].to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError("columns must contain numeric values") from exc
    if not np.isfinite(x).all():
        raise ValueError("columns must contain finite numeric values")
    if unlog2:
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.exp2(x)
        if unPlus1:
            x = x - 1
        if not np.isfinite(x).all():
            raise ValueError("unlog2 transformation produced non-finite values; input values are out of range")
        x = np.clip(x, a_min=0, a_max=None)
    else:
        x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("columns must contain non-negative expression values")
    return x


def calc_tau(
    df: pd.DataFrame, columns: str | Sequence[str], unlog2: bool = True, unPlus1: bool = True
) -> NDArray[np.float64]:
    """Calculate the standard tissue-specificity tau index for each row.

    Tau is ``sum(1 - x_i / max(x)) / (n - 1)`` for two or more tissues.
    A single-tissue profile and an all-zero profile are defined as ``0`` because
    tissue specificity cannot be inferred in those cases.

    Defaults assume log2(expression + 1): apply ``2**x - 1`` and clip at zero.
    Use ``unlog2=False`` for raw non-negative expression; ``unPlus1`` is ignored
    in that mode. For log2(expression), use ``unlog2=True, unPlus1=False``.
    Selected values must be finite. The input dataframe is not modified.
    """
    for flag_name, flag_value in [("unlog2", unlog2), ("unPlus1", unPlus1)]:
        if not isinstance(flag_value, (bool, np.bool_)):
            raise ValueError(f"{flag_name} must be a boolean value")
    columns = _validate_tau_columns(df, columns)
    x = _prepare_tau_matrix(df, columns, bool(unlog2), bool(unPlus1))
    if x.shape[1] == 1:
        return np.zeros(x.shape[0], dtype=float)
    xmax = x.max(axis=1).reshape(x.shape[0], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            x,
            xmax,
            out=np.full_like(x, np.nan, dtype=float),
            where=(xmax != 0),
        )
    xadj = 1 - ratio
    xadj = np.nan_to_num(xadj)
    taus = xadj.sum(axis=1) / (x.shape[1] - 1)
    return taus
