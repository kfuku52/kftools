"""Shared validation helpers used across :mod:`kftools` modules."""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def validate_boolean_flag(flag_value: Any, argument_name: str) -> bool:
    """Return a validated boolean, accepting NumPy boolean scalars."""
    if not isinstance(flag_value, (bool, np.bool_)):
        raise ValueError(f"{argument_name} must be a boolean value")
    return bool(flag_value)


def is_hashable(value: Any) -> bool:
    """Return whether *value* can safely be used as a mapping key."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


def coerce_path_argument(path_value: Any, argument_name: str = "file") -> str:
    """Return a text filesystem path or raise a field-specific ``ValueError``."""
    if isinstance(path_value, (bytes, bytearray)):
        raise ValueError(f"{argument_name} must be a path-like string (bytes are not supported)")
    if not isinstance(path_value, (str, os.PathLike)):
        raise ValueError(f"{argument_name} must be a path-like string")
    try:
        coerced_path = os.fspath(path_value)
    except TypeError as exc:
        raise ValueError(f"{argument_name} must be a path-like string") from exc
    if isinstance(coerced_path, (bytes, bytearray)):
        raise ValueError(f"{argument_name} must be a path-like string (bytes are not supported)")
    return coerced_path
