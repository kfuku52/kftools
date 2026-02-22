import numpy as np

def calc_complementarity(array1, array2):
    try:
        arr1 = np.asarray(array1, dtype=float).reshape(-1)
        arr2 = np.asarray(array2, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("array1 and array2 must contain numeric values") from exc
    if (not np.isfinite(arr1).all()) or (not np.isfinite(arr2).all()):
        raise ValueError("array1 and array2 must contain only finite numeric values")
    denom = arr1.size
    if denom == 0:
        raise ValueError("array1 must contain at least one value")
    n = min(arr1.size, arr2.size)
    if n == 0:
        return 0.0
    arr1 = arr1[:n]
    arr2 = arr2[:n]
    max_values = np.maximum(arr1, arr2)
    abs_diff = np.abs(arr1 - arr2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = np.divide(abs_diff, max_values, out=np.zeros_like(abs_diff), where=(max_values != 0))
    normalized_dif = rel_diff.sum() / denom
    return float(normalized_dif)


def calc_tau(df, columns, unlog2=True, unPlus1=True):
    if not hasattr(df, "columns"):
        raise ValueError("df must be a pandas DataFrame-like object with columns")
    for flag_name, flag_value in [("unlog2", unlog2), ("unPlus1", unPlus1)]:
        if not isinstance(flag_value, (bool, np.bool_)):
            raise ValueError(f"{flag_name} must be a boolean value")
    unlog2 = bool(unlog2)
    unPlus1 = bool(unPlus1)
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
    try:
        x = df.loc[:, columns].to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError("columns must contain numeric values") from exc
    if not np.isfinite(x).all():
        raise ValueError("columns must contain finite numeric values")
    if unlog2:
        with np.errstate(over='ignore', invalid='ignore'):
            x = np.exp2(x)
        if unPlus1:
            x = x - 1
        if not np.isfinite(x).all():
            raise ValueError("unlog2 transformation produced non-finite values; input values are out of range")
        x = np.clip(x, a_min=0, a_max=None)
    else:
        x = np.asarray(x, dtype=float)
    xmax = x.max(axis=1).reshape(x.shape[0], 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(
            x,
            xmax,
            out=np.full_like(x, np.nan, dtype=float),
            where=(xmax != 0),
        )
    xadj = 1 - ratio
    xadj = np.nan_to_num(xadj)
    taus = xadj.sum(axis=1) / x.shape[1]
    return taus
