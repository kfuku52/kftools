import numpy


def add_dict_key_prefix(d, prefix):
    return {prefix + '_' + key: value for key, value in d.items()}


def rgb_to_hex(r, g, b):
    rgb = [r, g, b]
    for i, channel in enumerate(rgb):
        assert channel <= 1
        rgb[i] = int(numpy.round(channel * 255, decimals=0))
    return '#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2])


def get_rgb_gradient(ncol, col1, col2, colm=None):
    if colm is None:
        colm = [0.5, 0.5, 0.5]
    if ncol <= 0:
        return []
    nmid = ncol / 2.0
    idx = numpy.arange(ncol, dtype=float)
    col1 = numpy.asarray(col1, dtype=float)
    col2 = numpy.asarray(col2, dtype=float)
    colm = numpy.asarray(colm, dtype=float)

    w1 = numpy.zeros(ncol, dtype=float)
    w2 = numpy.zeros(ncol, dtype=float)
    wm = numpy.ones(ncol, dtype=float)
    if nmid != 0:
        left = idx < nmid
        right = idx > nmid
        mid = ~(left | right)
        w1[left] = (nmid - idx[left]) / nmid
        wm[left] = 1 - w1[left]
        w2[right] = (idx[right] - nmid) / nmid
        wm[right] = 1 - w2[right]
        wm[mid] = 1.0
    cols = (w1[:, None] * col1[None, :]) + (wm[:, None] * colm[None, :]) + (w2[:, None] * col2[None, :])
    return cols.tolist()
