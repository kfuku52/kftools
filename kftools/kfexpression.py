import numpy

def calc_complementarity(array1, array2):
    sum_dif = float()
    for item1, item2 in zip(array1, array2):
        if (item1 == item2):
            sum_dif += 0
        else:
            if item1 > item2:
                sum_dif += (item1 - item2) / item1
            else:
                sum_dif += (item2 - item1) / item2
    normalized_dif = sum_dif / len(array1)
    return (normalized_dif)

def calc_tau(df, columns, unlog2=True, unPlus1=True):
    if unlog2:
        x = numpy.exp2(df.loc[:, columns])
        if unPlus1:
            x = x - 1
        x = x.clip(lower=0).values
    else:
        x = df.loc[:, columns]
    xmax = x.max(axis=1).reshape(x.shape[0], 1)
    xadj = 1 - (x / xmax)
    xadj = numpy.nan_to_num(xadj)
    taus = xadj.sum(axis=1) / x.shape[1]
    return taus