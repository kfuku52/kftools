import numpy


def calc_complementarity(array1, array2):
    arr1 = numpy.asarray(array1, dtype=float).reshape(-1)
    arr2 = numpy.asarray(array2, dtype=float).reshape(-1)
    denom = arr1.size
    if denom == 0:
        raise ZeroDivisionError("array1 must contain at least one value")
    n = min(arr1.size, arr2.size)
    if n == 0:
        return 0.0
    arr1 = arr1[:n]
    arr2 = arr2[:n]
    max_values = numpy.maximum(arr1, arr2)
    abs_diff = numpy.abs(arr1 - arr2)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        rel_diff = numpy.divide(abs_diff, max_values, out=numpy.zeros_like(abs_diff), where=(max_values != 0))
    normalized_dif = rel_diff.sum() / denom
    return float(normalized_dif)


def calc_tau(df, columns, unlog2=True, unPlus1=True):
    if unlog2:
        x = numpy.exp2(df.loc[:, columns])
        if unPlus1:
            x = x - 1
        x = x.clip(lower=0).values
    else:
        x = df.loc[:, columns].values
    xmax = x.max(axis=1).reshape(x.shape[0], 1)
    xadj = 1 - (x / xmax)
    xadj = numpy.nan_to_num(xadj)
    taus = xadj.sum(axis=1) / x.shape[1]
    return taus
