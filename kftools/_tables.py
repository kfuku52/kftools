"""Dataframe validation, deltas, and indexed ancestor traversal.

The public entry points are re-exported from :mod:`kftools.kfog`.
"""

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._validation import is_hashable


def _validate_column_name(column_name, argument_name):
    if not isinstance(column_name, str):
        raise ValueError(f"{argument_name} must be a string column name")
    if column_name.strip() == "":
        raise ValueError(f"{argument_name} must not be an empty string")


def _validate_hashable_series_values(series, argument_name):
    non_missing_values = series.dropna().to_list()
    unhashable_examples = []
    for value in non_missing_values:
        if not is_hashable(value):
            unhashable_examples.append(str(value))
            if len(unhashable_examples) >= 5:
                break
    if len(unhashable_examples) > 0:
        raise ValueError(f"{argument_name} must contain hashable values; invalid examples: {unhashable_examples}")


def _validate_non_missing_series_values(series, argument_name):
    missing_mask = series.isna()
    if missing_mask.any():
        raise ValueError(f"{argument_name} must not contain missing values")


def _is_missing_scalar(value):
    """Return whether a scalar is missing without triggering pd.NA truthiness."""
    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _scalar_values_equal(left, right):
    """Compare scalar values while treating two missing values as equal."""
    left_missing = _is_missing_scalar(left)
    right_missing = _is_missing_scalar(right)
    if left_missing or right_missing:
        return left_missing and right_missing
    try:
        comparison = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(comparison, (bool, np.bool_)) and bool(comparison)


def compute_delta(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Copy a frame and set delta_<column> to child-minus-parent values.

    Branch IDs must be unique across the frame; group reused IDs first.
    Missing parents/values yield NaN. Row order, index, and unrelated columns
    are preserved; the selected column is converted to numeric values.
    """
    if not hasattr(df, "columns"):
        raise ValueError("compute_delta requires a dataframe-like input with columns")
    _validate_column_name(column, "column")
    required_columns = {"branch_id", "parent", column}
    missing_columns = sorted(required_columns - set(df.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"compute_delta requires columns: {missing_columns}")
    out = df.copy()
    _validate_non_missing_series_values(out["branch_id"], "compute_delta branch_id column")
    _validate_hashable_series_values(out["branch_id"], "compute_delta branch_id column")
    _validate_hashable_series_values(out["parent"], "compute_delta parent column")
    if not out["branch_id"].is_unique:
        raise ValueError("compute_delta requires unique branch_id values")
    numeric_column = pd.to_numeric(out[column], errors="coerce")
    invalid_numeric_mask = out[column].notna() & numeric_column.isna()
    if invalid_numeric_mask.any():
        invalid_values = sorted(set(out.loc[invalid_numeric_mask, column].astype(str)))
        raise ValueError(
            f"compute_delta requires numeric values in column '{column}'; invalid values: {invalid_values}"
        )
    non_finite_mask = numeric_column.notna() & (~np.isfinite(numeric_column.to_numpy(dtype=float, copy=False)))
    if non_finite_mask.any():
        invalid_values = sorted(set(out.loc[non_finite_mask, column].astype(str)))
        raise ValueError(
            f"compute_delta requires finite numeric values in column '{column}'; invalid values: {invalid_values}"
        )
    out[column] = numeric_column
    value_by_label = out.set_index("branch_id")[column]
    parent_values = out["parent"].map(value_by_label)
    out[f"delta_{column}"] = out[column] - parent_values
    return out


def _validate_hashable_scalar(value, message):
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(message) from exc


def _most_recent_table(b, og, target_col, return_col, og_col):
    required_columns = {"branch_id", "parent", target_col, return_col, og_col}
    missing_columns = sorted(required_columns - set(b.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"get_most_recent requires columns: {missing_columns}")
    columns = list(dict.fromkeys(["branch_id", "parent", target_col, return_col]))
    b_og = b.loc[b[og_col] == og, columns]
    _validate_non_missing_series_values(b_og["branch_id"], "get_most_recent branch_id column")
    _validate_hashable_series_values(b_og["branch_id"], "get_most_recent branch_id column")
    _validate_hashable_series_values(b_og["parent"], "get_most_recent parent column")
    return b_og.drop_duplicates(subset="branch_id", keep="first").set_index("branch_id", drop=False)


def _walk_most_recent(b_og, nl, target_col, target_value, return_col):
    current_nl = nl
    visited_nl = set()
    while True:
        if current_nl in visited_nl:
            return np.nan
        if current_nl not in b_og.index:
            return np.nan
        visited_nl.add(current_nl)
        current_value = b_og.at[current_nl, target_col]
        if _scalar_values_equal(current_value, target_value):
            return b_og.at[current_nl, return_col]
        current_parent = b_og.at[current_nl, "parent"]
        if pd.isna(current_parent):
            return np.nan
        current_nl = current_parent


@dataclass(frozen=True)
class MostRecentLookup:
    """Prepared orthogroup tables for repeated nearest-ancestor lookups."""

    target_col: str
    return_col: str
    og_col: str
    tables: dict[object, pd.DataFrame]

    def find(self, nl: Hashable, og: Hashable, target_value: object) -> object:
        """Search this node, then its ancestors, using the prepared indexes."""
        _validate_hashable_scalar(nl, "nl must be a hashable scalar branch_id value")
        _validate_hashable_scalar(og, "og must be a hashable value comparable to the orthogroup column")
        b_og = self.tables.get(og)
        if b_og is None or (nl not in b_og.index):
            return np.nan
        return _walk_most_recent(b_og, nl, self.target_col, target_value, self.return_col)


def prepare_most_recent_lookup(
    b: pd.DataFrame,
    target_col: str,
    return_col: str,
    og_col: str = "orthogroup",
) -> MostRecentLookup:
    """Prepare indexes once for repeated :func:`get_most_recent` operations.

    Duplicate branch IDs within each orthogroup keep the first row. Rebuild
    after changing source data; arbitrary objects in cells are not deep-copied.
    """
    if not hasattr(b, "columns"):
        raise ValueError("prepare_most_recent_lookup requires a dataframe-like input with columns")
    _validate_column_name(target_col, "target_col")
    _validate_column_name(return_col, "return_col")
    _validate_column_name(og_col, "og_col")
    required_columns = {"branch_id", "parent", target_col, return_col, og_col}
    missing_columns = sorted(required_columns - set(b.columns))
    if missing_columns:
        raise ValueError(f"prepare_most_recent_lookup requires columns: {missing_columns}")

    columns = list(dict.fromkeys(["branch_id", "parent", target_col, return_col, og_col]))
    prepared_source = b.loc[:, columns]
    _validate_non_missing_series_values(prepared_source["branch_id"], "prepare_most_recent_lookup branch_id column")
    _validate_hashable_series_values(prepared_source["branch_id"], "prepare_most_recent_lookup branch_id column")
    _validate_hashable_series_values(prepared_source["parent"], "prepare_most_recent_lookup parent column")
    _validate_hashable_series_values(prepared_source[og_col], "prepare_most_recent_lookup orthogroup column")

    tables: dict[object, pd.DataFrame] = {}
    table_columns = list(dict.fromkeys(["branch_id", "parent", target_col, return_col]))
    for og_value, group in prepared_source.dropna(subset=[og_col]).groupby(og_col, sort=False, observed=True):
        tables[og_value] = (
            group.loc[:, table_columns]
            .drop_duplicates(subset="branch_id", keep="first")
            .set_index("branch_id", drop=False)
        )
    return MostRecentLookup(target_col, return_col, og_col, tables)


def get_most_recent(
    b: pd.DataFrame,
    nl: Hashable,
    og: Hashable,
    target_col: str,
    target_value: object,
    return_col: str,
    og_col: str = "orthogroup",
) -> object:
    """Return the nearest node value on the nl->root path matching a target state.

    The starting node is included. If no match is found before a missing node,
    missing parent, or cycle, return np.nan. Duplicate branch IDs within the
    orthogroup keep the first row; missing target values match each other.
    """
    if not hasattr(b, "columns"):
        raise ValueError("get_most_recent requires a dataframe-like input with columns")
    _validate_column_name(target_col, "target_col")
    _validate_column_name(return_col, "return_col")
    _validate_column_name(og_col, "og_col")
    _validate_hashable_scalar(nl, "nl must be a hashable scalar branch_id value")
    _validate_hashable_scalar(og, "og must be a hashable value comparable to the orthogroup column")
    b_og = _most_recent_table(b, og, target_col, return_col, og_col)
    if b_og.empty or (nl not in b_og.index):
        return np.nan
    return _walk_most_recent(b_og, nl, target_col, target_value, return_col)
