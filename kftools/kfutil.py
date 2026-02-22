import numpy as np
from collections.abc import Mapping


def add_dict_key_prefix(d, prefix):
    if not isinstance(d, Mapping):
        raise ValueError("d must be a mapping type (e.g., dict)")
    if not isinstance(prefix, str):
        raise ValueError("prefix must be a string")
    return {f"{prefix}_{key}": value for key, value in d.items()}


def rgb_to_hex(r, g, b):
    rgb = [r, g, b]
    for i, channel in enumerate(rgb):
        if isinstance(channel, bool) or (not isinstance(channel, (int, float, np.integer, np.floating))):
            raise ValueError("RGB channel values must be numeric")
        if not np.isfinite(channel):
            raise ValueError("RGB channel values must be finite")
        if (channel < 0) or (channel > 1):
            raise ValueError("RGB channel values must be between 0 and 1")
        rgb[i] = int(np.round(channel * 255, decimals=0))
    return '#%02X%02X%02X' % (rgb[0], rgb[1], rgb[2])


def get_rgb_gradient(ncol, col1, col2, colm=None):
    if isinstance(ncol, bool) or (not isinstance(ncol, (int, np.integer))):
        raise ValueError("ncol must be an integer")
    if colm is None:
        colm = [0.5, 0.5, 0.5]
    if ncol <= 0:
        return []
    color_inputs = {"col1": col1, "col2": col2, "colm": colm}
    for color_name, color_values in color_inputs.items():
        try:
            raw_color_arr = np.asarray(color_values, dtype=object).reshape(-1)
        except Exception as exc:
            raise ValueError(f"{color_name} must contain exactly 3 channel values") from exc
        if any(isinstance(channel_value, (bool, np.bool_)) for channel_value in raw_color_arr):
            raise ValueError(f"{color_name} channel values must be numeric (bool is not allowed)")
        try:
            color_arr = np.asarray(color_values, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{color_name} must contain exactly 3 channel values") from exc
        if color_arr.shape[0] != 3:
            raise ValueError(f"{color_name} must contain exactly 3 channel values")
        if (not np.isfinite(color_arr).all()) or (np.any(color_arr < 0)) or (np.any(color_arr > 1)):
            raise ValueError(f"{color_name} channel values must be finite numbers between 0 and 1")
    col1 = np.asarray(col1, dtype=float)
    col2 = np.asarray(col2, dtype=float)
    colm = np.asarray(colm, dtype=float)
    if ncol == 1:
        return [col1.tolist()]
    t_values = np.linspace(0.0, 1.0, num=ncol, endpoint=True)
    cols = np.empty((ncol, 3), dtype=float)
    left = t_values <= 0.5
    right = ~left
    if left.any():
        w = (t_values[left] / 0.5).reshape(-1, 1)
        cols[left, :] = ((1.0 - w) * col1.reshape(1, 3)) + (w * colm.reshape(1, 3))
    if right.any():
        w = ((t_values[right] - 0.5) / 0.5).reshape(-1, 1)
        cols[right, :] = ((1.0 - w) * colm.reshape(1, 3)) + (w * col2.reshape(1, 3))
    return cols.tolist()
