import copy
import gzip
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._validation import coerce_path_argument, is_hashable, validate_boolean_flag
from .kfexpression import calc_complementarity, calc_tau
from .kfphylo import (
    add_numerical_node_labels,
    check_ultrametric,
    load_phylo_tree,
    taxonomic_annotation,
)
from .kfspecies import parse_species_label

NOTUNG_OPT_ROOT_RE = re.compile(
    r"Number of optimal roots:\s*([0-9][0-9,]*)\s*out of\s*([0-9][0-9,]*)",
    flags=re.IGNORECASE,
)
NOTUNG_BEST_SCORE_RE = re.compile(
    r"Best rooting score:\s*(.*?)\s*,\s*worst rooting score:\s*(.*?)\s*$",
    flags=re.IGNORECASE,
)
ROOT_RETURNING_RE = re.compile(r"Returning the (.*) tree", flags=re.IGNORECASE)
ROOT_POSITIONS_RE = re.compile(r"root positions with rho peak:\s*(.*)", flags=re.IGNORECASE)
NOTUNG_DUP_RE = re.compile(r"-\s*Duplications\s*:\s*([0-9][0-9,]*)", flags=re.IGNORECASE)
NOTUNG_CODIV_RE = re.compile(r"-\s*Co[- ]?Divergences\s*:\s*([0-9][0-9,]*)", flags=re.IGNORECASE)
NOTUNG_TRANSFER_RE = re.compile(r"-\s*Transfers\s*:\s*([0-9][0-9,]*)", flags=re.IGNORECASE)
NOTUNG_LOSS_RE = re.compile(r"-\s*Losses\s*:\s*([0-9][0-9,]*)", flags=re.IGNORECASE)
NOTUNG_POLYTOMY_RE = re.compile(r"-\s*Polytomies\s*:\s*([0-9][0-9,]*)", flags=re.IGNORECASE)
INT64_MAX = np.iinfo(np.int64).max


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


def _normalize_locale_mantissa(mantissa):
    if ("," in mantissa) and ("." in mantissa):
        if mantissa.rfind(".") > mantissa.rfind(","):
            # 1,234.56 -> dot decimal, comma thousands
            mantissa = mantissa.replace(",", "")
        else:
            # 1.234,56 -> comma decimal, dot thousands
            mantissa = mantissa.replace(".", "").replace(",", ".")
    elif "," in mantissa:
        comma_parts = mantissa.split(",")
        if (
            (len(comma_parts) > 2)
            and all(len(part) == 3 for part in comma_parts[1:])
            and (comma_parts[0] not in ("", "+", "-"))
        ):
            # 1,234,567 -> comma thousands (multi-separator form)
            mantissa = "".join(comma_parts)
        else:
            # 1234,56 -> comma decimal
            mantissa = mantissa.replace(",", ".")
    elif "." in mantissa:
        dot_parts = mantissa.split(".")
        if (
            (len(dot_parts) > 2)
            and all(len(part) == 3 for part in dot_parts[1:])
            and (dot_parts[0] not in ("", "+", "-"))
        ):
            # 1.234.567 -> dot thousands (multi-separator form)
            mantissa = "".join(dot_parts)
    return mantissa


def _parse_float_locale(value):
    text = str(value).strip().replace(" ", "")
    if text == "":
        raise ValueError("empty float token")
    try:
        return float(text)
    except ValueError:
        pass
    split_match = re.match(r"^(.*?)([eE][-+]?[0-9]+)$", text)
    has_exponent_marker = ("e" in text) or ("E" in text)
    if has_exponent_marker and split_match is None:
        raise ValueError(f"invalid float token: {value}")
    mantissa = split_match.group(1) if split_match is not None else text
    exponent_part = split_match.group(2) if split_match is not None else ""
    mantissa = _normalize_locale_mantissa(mantissa)

    return float(mantissa + exponent_part)


def _nwk_age_values(tree, n_nodes):
    if not check_ultrametric(tree):
        raise ValueError("Tree must be ultrametric when age=True and attr='dist'")
    age_values = np.empty(n_nodes, dtype=float)
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf:
            age_values[node.branch_id] = 0.0
        else:
            first_child = node.children[0]
            age_values[node.branch_id] = age_values[first_child.branch_id] + first_child.dist
    return age_values


def _nwk_attr_values(nodes, attr):
    n_nodes = len(nodes)
    attr_values_raw = [np.nan] * n_nodes
    for node in nodes:
        attr_values_raw[node.branch_id] = getattr(node, attr, np.nan)
    # Let pandas infer a common dtype from all values.  Deriving the dtype from
    # one node can silently truncate mixed numeric attributes (e.g. 1.5 -> 1).
    return (
        np.asarray(attr_values_raw, dtype=object)
        if any(value is None for value in attr_values_raw)
        else attr_values_raw
    )


def _nwk_relation_values(nodes, parent, sister):
    n_nodes = len(nodes)
    parent_values = np.full(n_nodes, -1, dtype=np.int64) if parent else None
    sister_values = np.full(n_nodes, -1, dtype=np.int64) if sister else None
    sisters_values = np.empty(n_nodes, dtype=object) if sister else None
    if sisters_values is not None:
        sisters_values.fill(())

    for node in nodes:
        label = node.branch_id
        if parent_values is not None and (not node.is_root):
            parent_values[label] = node.up.branch_id
        if sister_values is not None and (not node.is_root):
            assert sisters_values is not None
            sister_labels = tuple(sister_node.branch_id for sister_node in node.get_sisters())
            sisters_values[label] = sister_labels
            if sister_labels:
                sister_values[label] = sister_labels[0]
    return parent_values, sister_values, sisters_values


def nwk2table(
    tree: Any,
    attr: str = "",
    age: bool = False,
    parent: bool = False,
    sister: bool = False,
) -> pd.DataFrame:
    """Convert a tree to a node table with optional ages and relationships.

    ``sister=True`` emits both the legacy first-sister ``sister`` column and a
    lossless tuple-valued ``sisters`` column for polytomies.
    """
    age = validate_boolean_flag(age, "age")
    parent = validate_boolean_flag(parent, "parent")
    sister = validate_boolean_flag(sister, "sister")
    if not isinstance(attr, str):
        raise ValueError("attr must be a string")
    if age and (attr != "dist"):
        raise ValueError("age=True is supported only when attr='dist'")
    if not hasattr(tree, "traverse"):
        try:
            tree = load_phylo_tree(tree, parser=0 if attr == "support" else 1)
        except TypeError as exc:
            raise ValueError("tree must be a Newick string, path, or tree object") from exc
    tree = add_numerical_node_labels(tree)
    nodes = list(tree.traverse())
    n_nodes = len(nodes)
    age_values = _nwk_age_values(tree, n_nodes) if age else None
    attr_values = _nwk_attr_values(nodes, attr)
    parent_values, sister_values, sisters_values = _nwk_relation_values(nodes, parent, sister)

    data: dict[str, object] = {"branch_id": np.arange(n_nodes, dtype=np.int64)}
    data[attr] = attr_values
    if age_values is not None:
        data["age"] = age_values
    if parent_values is not None:
        data["parent"] = parent_values
    if sister_values is not None:
        data["sister"] = sister_values
        data["sisters"] = sisters_values
    df = pd.DataFrame(data)
    return df


def _resolve_species_parser_alias(species_parser, parser):
    if parser is not None:
        if species_parser is not None:
            raise ValueError("Use only one of species_parser or parser")
        return parser
    return species_parser


def _load_og_tree(tree, argument_name):
    if hasattr(tree, "traverse"):
        return tree
    try:
        return load_phylo_tree(tree, parser=1)
    except TypeError as exc:
        raise ValueError(f"{argument_name} must be a Newick string, path, or tree object") from exc


def _species_tree_labels(species_tree, species_parser):
    invalid_names = [name for name in species_tree.leaf_names() if (not isinstance(name, str)) or (name.strip() == "")]
    if invalid_names:
        raise ValueError("species_tree leaf names must be non-empty strings for node_gene2species")
    label_by_leaf = {}
    name_counts: dict[str, int] = {}
    for leaf in species_tree.leaves():
        try:
            label = parse_species_label(leaf.name, species_parser=species_parser).species_label
        except ValueError as exc:
            raise ValueError(f"species_tree leaf name must contain species information with '_': {leaf.name}") from exc
        label_by_leaf[leaf] = label
        name_counts[label] = name_counts.get(label, 0) + 1
    duplicates = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"species_tree leaf names must be unique for node_gene2species; duplicates: {duplicates}")
    return label_by_leaf, name_counts


def _validate_gene_species_ultrametric(gene_tree, species_tree, is_ultrametric):
    if not is_ultrametric:
        return
    if not check_ultrametric(gene_tree):
        raise ValueError("gene_tree must be ultrametric when is_ultrametric=True")
    try:
        species_is_ultrametric = check_ultrametric(species_tree)
    except ValueError as exc:
        raise ValueError(
            "species_tree must be ultrametric with finite non-negative branch lengths when is_ultrametric=True"
        ) from exc
    if not species_is_ultrametric:
        raise ValueError("species_tree must be ultrametric when is_ultrametric=True")


def _gene_species_label(leaf_name, species_parser):
    try:
        return parse_species_label(leaf_name, species_parser=species_parser).species_label
    except ValueError as exc:
        raise ValueError(f"Gene leaf name must contain species information with '_': {leaf_name}") from exc


def _rename_gene_leaves(gene_tree, species_parser):
    for leaf in gene_tree.leaves():
        leaf.name = _gene_species_label(leaf.name, species_parser)


def _species_tree_context(species_tree, label_by_leaf, is_ultrametric):
    species_nodes = list(species_tree.traverse(strategy="postorder"))
    names = {
        node: (label_by_leaf[node] if node.is_leaf else (node.name or "")).replace("'", "") for node in species_nodes
    }
    leaf_node = {label_by_leaf[leaf]: leaf for leaf in species_tree.leaves()}
    species_depth: dict[object, int] = {}
    for node in species_tree.traverse(strategy="preorder"):
        species_depth[node] = 0 if node.is_root else species_depth[node.up] + 1
    age: dict[object, float] = {}
    up_age: dict[object, float] = {}
    if is_ultrametric:
        for node in species_nodes:
            age[node] = 0.0 if node.is_leaf else age[node.children[0]] + node.children[0].dist
        up_age = {node: np.inf if node.is_root else age[node.up] for node in species_nodes}
    return names, leaf_node, species_depth, age, up_age


def _pair_lca(node_a, node_b, depth, cache):
    if node_a is node_b:
        return node_a
    key = (node_a, node_b) if id(node_a) <= id(node_b) else (node_b, node_a)
    if key in cache:
        return cache[key]
    a, b = node_a, node_b
    depth_a, depth_b = depth[a], depth[b]
    while depth_a > depth_b:
        a, depth_a = a.up, depth_a - 1
    while depth_b > depth_a:
        b, depth_b = b.up, depth_b - 1
    while a is not b:
        a, b = a.up, b.up
    cache[key] = a
    return a


def _gene_coverage_context(gene_tree, species_leaf_node, species_depth, is_ultrametric):
    gene_nodes = list(gene_tree.traverse(strategy="postorder"))
    gene_coverage = {}
    gene_has_missing_species = {}
    gene_age = {}
    lca_cache: dict[tuple[object, object], object] = {}
    for gn in gene_nodes:
        if gn.is_leaf:
            covered_species_node = species_leaf_node.get(gn.name)
            gene_coverage[gn] = covered_species_node
            gene_has_missing_species[gn] = covered_species_node is None
            if is_ultrametric:
                gene_age[gn] = 0.0
            continue

        children = gn.children
        if is_ultrametric:
            first_child = children[0]
            gene_age[gn] = gene_age[first_child] + first_child.dist
        has_missing_species = any(gene_has_missing_species[child] for child in children)
        gene_has_missing_species[gn] = has_missing_species
        if has_missing_species:
            gene_coverage[gn] = None
            continue

        covered_species_node = gene_coverage[children[0]]
        for child in children[1:]:
            covered_species_node = _pair_lca(covered_species_node, gene_coverage[child], species_depth, lca_cache)
        gene_coverage[gn] = covered_species_node
    return gene_nodes, gene_coverage, gene_age


def _species_node_at_age(coverage_node, gene_node_age, species_age, species_up_age):
    current = coverage_node
    while current is not None:
        if species_age[current] <= gene_node_age < species_up_age[current]:
            return current
        current = current.up
    return None


def _gene_species_rows(context, is_ultrametric):
    gene_nodes, coverage, gene_age, species_names, species_age, species_up_age = context
    rows = []
    for gn in gene_nodes:
        coverage_node = coverage[gn]
        row = {
            "branch_id": gn.branch_id,
            "spnode_coverage": "" if coverage_node is None else species_names[coverage_node],
        }
        if is_ultrametric:
            row["spnode_age"] = ""
            if coverage_node is not None:
                age_node = _species_node_at_age(coverage_node, gene_age[gn], species_age, species_up_age)
                if age_node is not None:
                    row["spnode_age"] = species_names[age_node]
        rows.append(row)
    return rows


def node_gene2species(
    gene_tree: Any,
    species_tree: Any,
    is_ultrametric: bool = False,
    species_parser: Any = None,
    parser: Any = None,
) -> pd.DataFrame:
    """Map each gene-tree node to its covered species-tree node."""
    is_ultrametric = validate_boolean_flag(is_ultrametric, "is_ultrametric")
    species_parser = _resolve_species_parser_alias(species_parser, parser)
    gene_tree = _load_og_tree(gene_tree, "gene_tree")
    species_tree = _load_og_tree(species_tree, "species_tree")
    label_by_leaf, species_counts = _species_tree_labels(species_tree, species_parser)
    gene_tree = add_numerical_node_labels(copy.deepcopy(gene_tree))
    _validate_gene_species_ultrametric(gene_tree, species_tree, is_ultrametric)
    _rename_gene_leaves(gene_tree, species_parser)
    missing_species = set(gene_tree.leaf_names()) - set(species_counts)
    if missing_species:
        warnings.warn(
            f"A total of {len(missing_species)} species are missing in the species tree: {sorted(missing_species)}",
            RuntimeWarning,
            stacklevel=2,
        )
    species_names, leaf_node, depth, species_age, species_up_age = _species_tree_context(
        species_tree, label_by_leaf, is_ultrametric
    )
    gene_nodes, coverage, gene_age = _gene_coverage_context(gene_tree, leaf_node, depth, is_ultrametric)
    rows = _gene_species_rows(
        (gene_nodes, coverage, gene_age, species_names, species_age, species_up_age),
        is_ultrametric,
    )
    columns = ["branch_id", "spnode_coverage", "spnode_age"] if is_ultrametric else ["branch_id", "spnode_coverage"]
    return pd.DataFrame(rows, columns=columns)


def _read_tsv(file, argument_name):
    try:
        return pd.read_csv(file, sep="\t")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Failed to read {argument_name} as UTF-8 tab-separated text: {file}") from exc


def _invalid_column_values(df, mask, column):
    return sorted(set(df.loc[mask, column].astype(str)))


def _validated_regime_series(df, df_name):
    numeric = pd.to_numeric(df["regime"], errors="coerce")
    validations = [
        (df["regime"].notna() & numeric.isna(), "must be numeric or NaN"),
        (
            numeric.notna() & (~np.isfinite(numeric.to_numpy(dtype=float, copy=False))),
            "must contain finite numeric values",
        ),
        (numeric.notna() & (numeric != np.floor(numeric)), "must contain integer IDs"),
        (numeric.notna() & (numeric < 0), "must contain non-negative IDs"),
        (numeric.notna() & (numeric > INT64_MAX), f"must be <= {INT64_MAX} to avoid integer overflow"),
    ]
    for invalid_mask, message in validations:
        if invalid_mask.any():
            invalid_values = _invalid_column_values(df, invalid_mask, "regime")
            prefix = f"{df_name} " if df_name else ""
            raise ValueError(f"{prefix}regime column {message}; invalid values: {invalid_values}")
    return numeric


def _validated_trait_columns(df_leaf, required_columns):
    trait_columns = [column for column in df_leaf.columns if column not in required_columns]
    if not trait_columns:
        raise ValueError("leaf_file must include at least one trait column after node_name/param/regime")
    for trait_column in trait_columns:
        numeric = pd.to_numeric(df_leaf[trait_column], errors="coerce")
        invalid = df_leaf[trait_column].notna() & numeric.isna()
        if invalid.any():
            invalid_values = _invalid_column_values(df_leaf, invalid, trait_column)
            raise ValueError(
                f"leaf_file trait column '{trait_column}' must be numeric or NaN; invalid values: {invalid_values}"
            )
        df_leaf[trait_column] = numeric
    return trait_columns


def _validated_ou_tables(df_regime, df_leaf):
    missing_regime = sorted({"node_name", "regime"} - set(df_regime.columns))
    if missing_regime:
        raise ValueError(f"regime_file requires columns: {missing_regime}")
    required_leaf = {"node_name", "param", "regime"}
    missing_leaf = sorted(required_leaf - set(df_leaf.columns))
    if missing_leaf:
        raise ValueError(f"leaf_file requires columns: {missing_leaf}")
    df_regime, df_leaf = df_regime.copy(), df_leaf.copy()
    df_regime["regime"] = _validated_regime_series(df_regime, "regime_file")
    df_leaf["regime"] = _validated_regime_series(df_leaf, "leaf_file")
    traits = _validated_trait_columns(df_leaf, required_leaf)
    return df_regime, df_leaf, traits


def _ou_regime_map(nodes, df_regime):
    rows = df_regime.loc[df_regime["regime"].notna(), ["node_name", "regime"]].copy()
    invalid = rows["node_name"].map(lambda name: (not isinstance(name, str)) or (name.strip() == ""))
    if invalid.any():
        values = _invalid_column_values(rows, invalid, "node_name")
        raise ValueError(
            "regime_file node_name column must contain non-empty string values when regime is provided; "
            f"invalid values: {values}"
        )
    conflicts = rows.groupby("node_name")["regime"].nunique(dropna=True)
    conflicting_names = sorted(conflicts.index[conflicts > 1].tolist())
    if conflicting_names:
        raise ValueError(f"regime_file contains conflicting regime IDs for node_name values: {conflicting_names}")
    named_nodes = [node.name for node in nodes if isinstance(node.name, str) and node.name.strip()]
    name_counts = pd.Series(named_nodes, dtype=object).value_counts()
    duplicate_names = sorted(name_counts.index[name_counts > 1].tolist())
    if duplicate_names:
        raise ValueError(
            "input_tree_file contains duplicate non-empty node names that make regime mapping ambiguous: "
            f"{duplicate_names}"
        )
    unknown_names = sorted(set(rows["node_name"]) - set(named_nodes))
    if unknown_names:
        raise ValueError(f"regime_file contains node_name values not present in input_tree_file: {unknown_names}")
    return {name: int(regime) for name, regime in rows.itertuples(index=False, name=None)}


def _assign_ou_regimes(tree, regime_map):
    for node in tree.traverse(strategy="preorder"):
        inherited = 0 if node.is_root else node.up.regime
        node.regime = regime_map.get(node.name, inherited)


def _ou_mu_by_regime(df_leaf, tissues, nodes):
    if "expectations" in df_leaf["param"].values:
        df_leaf.loc[df_leaf["param"] == "expectations", "param"] = "mu"
    columns = [column for column in df_leaf.columns if column not in ["node_name", "param"]]
    mu_table = df_leaf.loc[df_leaf["param"] == "mu", columns].drop_duplicates().groupby("regime").mean().loc[:, tissues]
    mu_by_regime = dict(zip(mu_table.index.to_numpy(), mu_table.to_numpy(), strict=True))
    missing = sorted({node.regime for node in nodes} - set(mu_by_regime))
    if missing:
        raise ValueError(f"Missing mu values for regime IDs: {missing}")
    return mu_by_regime


def _ou_node_arrays(nodes, mu_by_regime, num_tissues):
    num_nodes = len(nodes)
    arrays = {
        "branch_id": np.empty(num_nodes, dtype=np.int64),
        "regime": np.empty(num_nodes, dtype=np.int64),
        "is_shift": np.empty(num_nodes, dtype=np.int64),
        "num_child_shift": np.full(num_nodes, np.nan, dtype=float),
        "parent_labels": np.empty(num_nodes, dtype=np.int64),
        "mu_values": np.empty((num_nodes, num_tissues), dtype=float),
    }
    shift_pairs = []
    for row_index, node in enumerate(nodes):
        shift = int((not node.is_root) and (node.regime != node.up.regime))
        arrays["branch_id"][row_index] = node.branch_id
        arrays["regime"][row_index] = node.regime
        arrays["is_shift"][row_index] = shift
        arrays["parent_labels"][row_index] = -1 if node.is_root else node.up.branch_id
        arrays["mu_values"][row_index, :] = mu_by_regime[node.regime]
        if not node.is_leaf:
            arrays["num_child_shift"][row_index] = sum(int(node.regime != child.regime) for child in node.children)
        sisters = node.get_sisters() if shift else []
        if sisters:
            shift_pairs.append((node.branch_id, sisters[0].branch_id))
    return arrays, shift_pairs


def _add_ou_tau_values(df, arrays, mu_columns):
    tau_values = calc_tau(df, mu_columns, unlog2=True, unPlus1=True)
    branch_id, parent_labels = arrays["branch_id"], arrays["parent_labels"]
    tau_by_label = np.empty(len(branch_id), dtype=float)
    tau_by_label[branch_id] = tau_values
    safe_parent = parent_labels.copy()
    safe_parent[safe_parent == -1] = 0
    delta_tau = tau_values - tau_by_label[safe_parent]
    delta_tau[parent_labels == -1] = np.nan
    df["tau"], df["delta_tau"] = tau_values, delta_tau


def _add_ou_shift_values(df, arrays, shift_pairs):
    branch_id, mu_values = arrays["branch_id"], arrays["mu_values"]
    label_to_idx = np.empty(len(branch_id), dtype=np.int64)
    label_to_idx[branch_id] = np.arange(len(branch_id), dtype=np.int64)
    delta_maxmu = np.full(len(branch_id), np.nan, dtype=float)
    complementarity = np.full(len(branch_id), np.nan, dtype=float)
    max_mu = mu_values.max(axis=1)
    unlogged_mu = np.clip(np.exp2(mu_values) - 1, a_min=0, a_max=None)
    for label, sister_label in shift_pairs:
        index, sister_index = label_to_idx[label], label_to_idx[sister_label]
        delta_maxmu[index] = float(max_mu[index] - max_mu[sister_index])
        complementarity[index] = calc_complementarity(unlogged_mu[index], unlogged_mu[sister_index])
    df["delta_maxmu"], df["mu_complementarity"] = delta_maxmu, complementarity


def ou2table(regime_file: Any, leaf_file: Any, input_tree_file: Any) -> pd.DataFrame:
    """Combine OU regime, leaf-parameter, and tree files into a node table."""
    regime_file = coerce_path_argument(regime_file, "regime_file")
    leaf_file = coerce_path_argument(leaf_file, "leaf_file")
    input_tree_file = coerce_path_argument(input_tree_file, "input_tree_file")
    if (not os.path.exists(input_tree_file)) or (not os.path.isfile(input_tree_file)):
        raise ValueError(f"input_tree_file must be an existing file path: {input_tree_file}")
    df_regime = _read_tsv(regime_file, "regime_file")
    df_leaf = _read_tsv(leaf_file, "leaf_file")
    df_regime, df_leaf, tissues = _validated_ou_tables(df_regime, df_leaf)
    tree = add_numerical_node_labels(load_phylo_tree(input_tree_file, parser=1))
    nodes = list(tree.traverse())
    cn1 = ["branch_id", "regime", "is_shift", "num_child_shift"]
    cn2 = ["tau", "delta_tau", "delta_maxmu", "mu_complementarity"]
    cn3 = ["mu_" + tissue for tissue in tissues]
    cn = cn1 + cn2 + cn3
    _assign_ou_regimes(tree, _ou_regime_map(nodes, df_regime))
    mu_by_regime = _ou_mu_by_regime(df_leaf, tissues, nodes)
    arrays, shift_pairs = _ou_node_arrays(nodes, mu_by_regime, len(cn3))
    df = pd.DataFrame({column: arrays[column] for column in ["branch_id", "regime", "is_shift", "num_child_shift"]})
    for col_idx, col in enumerate(cn3):
        df[col] = arrays["mu_values"][:, col_idx]
    _add_ou_tau_values(df, arrays, cn3)
    _add_ou_shift_values(df, arrays, shift_pairs)
    return df.loc[:, cn]


def _annotate_misc_taxonomy(tree, nodes, tax_annot, species_parser):
    if tax_annot:
        return taxonomic_annotation(tree, species_parser=species_parser)
    for node in nodes:
        node.taxid = -999
        if not node.is_leaf:
            node.sci_name = ""
            continue
        try:
            node.sci_name = parse_species_label(node.name, species_parser=species_parser).scientific_name
        except ValueError:
            node.sci_name = node.name
    return tree


def _dup_confidence_score(children, species_masks):
    if len(children) < 2:
        return 0.0
    seen_once = 0
    seen_multiple = 0
    for child in children:
        child_mask = species_masks[child]
        seen_multiple |= seen_once & child_mask
        seen_once |= child_mask
    union_count = seen_once.bit_count()
    return 0.0 if union_count == 0 else seen_multiple.bit_count() / union_count


def _misc_descendant_statistics(tree):
    species_index: dict[str, int] = {}
    species_mask_by_node: dict[object, int] = {}
    num_leaf_by_node: dict[object, int] = {}
    dup_conf_score_by_node: dict[object, float] = {}
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf:
            species_id = species_index.setdefault(node.sci_name, len(species_index))
            species_mask_by_node[node] = 1 << species_id
            num_leaf_by_node[node] = 1
            dup_conf_score_by_node[node] = 0.0
            continue
        children = node.children
        species_mask = 0
        descendant_leaf_count = 0
        for child in children:
            species_mask |= species_mask_by_node[child]
            descendant_leaf_count += num_leaf_by_node[child]
        species_mask_by_node[node] = species_mask
        num_leaf_by_node[node] = descendant_leaf_count
        dup_conf_score_by_node[node] = _dup_confidence_score(children, species_mask_by_node)
    return species_mask_by_node, num_leaf_by_node, dup_conf_score_by_node


def _misc_output_arrays(n_nodes):
    children = np.empty(n_nodes, dtype=object)
    children.fill(())
    sisters = np.empty(n_nodes, dtype=object)
    sisters.fill(())
    return {
        "branch_id": np.empty(n_nodes, dtype=np.int64),
        "taxon": np.empty(n_nodes, dtype=object),
        "taxid": np.empty(n_nodes, dtype=np.int64),
        "num_sp": np.empty(n_nodes, dtype=np.int64),
        "num_leaf": np.empty(n_nodes, dtype=np.int64),
        "so_event": np.full(n_nodes, "L", dtype=object),
        "dup_conf_score": np.zeros(n_nodes, dtype=float),
        "parent": np.full(n_nodes, -999, dtype=np.int64),
        "sister": np.full(n_nodes, -999, dtype=np.int64),
        "sisters": sisters,
        "child1": np.full(n_nodes, -999, dtype=np.int64),
        "child2": np.full(n_nodes, -999, dtype=np.int64),
        "children": children,
        "so_event_parent": np.full(n_nodes, "S", dtype=object),
    }


def _misc_sister_label(node):
    if node.up is None:
        return -999
    siblings = node.up.children
    if len(siblings) == 2:
        return (siblings[1] if siblings[0] is node else siblings[0]).branch_id
    sisters = node.get_sisters()
    return sisters[0].branch_id if sisters else -999


def _fill_misc_node_row(arrays, row_idx, node, statistics):
    species_masks, leaf_counts, dup_scores = statistics
    arrays["branch_id"][row_idx] = node.branch_id
    arrays["taxon"][row_idx] = str(node.sci_name)
    arrays["taxid"][row_idx] = node.taxid
    arrays["num_sp"][row_idx] = species_masks[node].bit_count()
    arrays["num_leaf"][row_idx] = leaf_counts[node]
    arrays["parent"][row_idx] = -999 if node.up is None else node.up.branch_id
    arrays["sister"][row_idx] = _misc_sister_label(node)
    arrays["sisters"][row_idx] = tuple(sister.branch_id for sister in node.get_sisters())
    arrays["children"][row_idx] = tuple(child.branch_id for child in node.children)
    if node.is_leaf:
        return
    if node.children:
        arrays["child1"][row_idx] = node.children[0].branch_id
    if len(node.children) >= 2:
        arrays["child2"][row_idx] = node.children[1].branch_id
    score = dup_scores.get(node, 0.0)
    arrays["dup_conf_score"][row_idx] = score
    arrays["so_event"][row_idx] = "D" if score > 0 else "S"


def _fill_misc_parent_events(arrays, nodes, dup_scores):
    for row_idx, node in enumerate(nodes):
        if (node.up is not None) and (dup_scores.get(node.up, 0.0) > 0):
            arrays["so_event_parent"][row_idx] = "D"


def get_misc_node_statistics(
    tree_file: Any,
    tax_annot: bool = False,
    species_parser: Any = None,
    parser: Any = None,
) -> pd.DataFrame:
    """Calculate descendant, event, taxonomy, and relationship statistics."""
    tax_annot = validate_boolean_flag(tax_annot, "tax_annot")
    species_parser = _resolve_species_parser_alias(species_parser, parser)
    tree = _load_og_tree(tree_file, "tree_file")
    tree = add_numerical_node_labels(tree)
    nodes = list(tree.traverse())
    tree = _annotate_misc_taxonomy(tree, nodes, tax_annot, species_parser)
    statistics = _misc_descendant_statistics(tree)
    arrays = _misc_output_arrays(len(nodes))
    for row_idx, node in enumerate(nodes):
        _fill_misc_node_row(arrays, row_idx, node, statistics)
    _fill_misc_parent_events(arrays, nodes, statistics[2])
    columns = [
        "branch_id",
        "taxon",
        "taxid",
        "num_sp",
        "num_leaf",
        "so_event",
        "dup_conf_score",
        "parent",
        "sister",
        "sisters",
        "child1",
        "child2",
        "children",
        "so_event_parent",
    ]
    return pd.DataFrame(arrays, columns=columns)


def compute_delta(df: Any, column: str) -> pd.DataFrame:
    """Return a dataframe copy with child-minus-parent values for ``column``."""
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
    parent_column = f"parent_{column}"
    value_by_label = out.set_index("branch_id")[column]
    out[parent_column] = out["parent"].map(value_by_label)
    out[f"delta_{column}"] = out[column] - out[parent_column]
    out = out.drop(parent_column, axis=1)
    return out


def get_notung_root_stats(file: Any) -> dict[str, Any]:
    """Parse optimal-root counts and root scores from a Notung log."""
    file = coerce_path_argument(file, "file")
    out = {}
    try:
        with open(file) as f:
            for line in f:
                m_opt = NOTUNG_OPT_ROOT_RE.search(line)
                if m_opt is not None:
                    out["ntg_num_opt_root"] = int(m_opt.group(1).replace(",", ""))
                m_best = NOTUNG_BEST_SCORE_RE.search(line)
                if m_best is not None:
                    try:
                        best_score = _parse_float_locale(m_best.group(1))
                        worst_score = _parse_float_locale(m_best.group(2))
                    except ValueError:
                        continue
                    out["ntg_best_root_score"] = best_score
                    out["ntg_worst_root_score"] = worst_score
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    return out


def get_notung_reconcil_stats(file: Any) -> dict[str, int]:
    """Parse reconciliation event counts from a Notung log."""
    file = coerce_path_argument(file, "file")
    out = {}
    count_patterns = [
        ("ntg_num_dup", NOTUNG_DUP_RE),
        ("ntg_num_codiv", NOTUNG_CODIV_RE),
        ("ntg_num_transfer", NOTUNG_TRANSFER_RE),
        ("ntg_num_loss", NOTUNG_LOSS_RE),
        ("ntg_num_polytomy", NOTUNG_POLYTOMY_RE),
    ]
    try:
        with open(file) as f:
            for line in f:
                for key, pattern in count_patterns:
                    m = pattern.search(line)
                    if m is not None:
                        out[key] = int(m.group(1).replace(",", ""))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    return out


def get_root_stats(file: Any) -> dict[str, Any]:
    """Parse rooting method and rho-peak counts from a rooting log."""
    file = coerce_path_argument(file, "file")
    out: dict[str, object] = {}
    try:
        with open(file) as f:
            for line in f:
                m_pos = ROOT_POSITIONS_RE.search(line)
                if m_pos is not None:
                    positions = m_pos.group(1).strip()
                    if positions == "":
                        out["num_rho_peak"] = 0
                    else:
                        tokens = [tok for tok in re.split(r"[\s,]+", positions) if tok != ""]
                        placeholder_tokens = {"-", "none", "na", "n/a", "null"}
                        valid_tokens = [tok for tok in tokens if tok.lower() not in placeholder_tokens]
                        out["num_rho_peak"] = len(valid_tokens)
                m = ROOT_RETURNING_RE.search(line)
                if m is not None:
                    rooting_method = m.group(1).strip()
                    if rooting_method.lower().startswith("first "):
                        rooting_method = rooting_method[6:].strip()
                    out["rooting_method"] = rooting_method
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    return out


def get_aln_stats(file: Any) -> dict[str, int]:
    """Return aligned-site, sequence-count, and ungapped-length FASTA stats."""
    file = coerce_path_argument(file, "file")
    out = {}
    seq_lens_w_gap = []
    seq_lens = []
    seq_w_gap_len = 0
    seq_len = 0
    has_sequence = False
    try:
        with open(file) as f:
            for line in f:
                if line.startswith(">"):
                    if has_sequence:
                        seq_lens_w_gap.append(seq_w_gap_len)
                        seq_lens.append(seq_len)
                    seq_w_gap_len = 0
                    seq_len = 0
                    has_sequence = True
                    continue
                seq_line = line.strip()
                if not seq_line:
                    continue
                if not has_sequence:
                    raise ValueError("alignment file must be FASTA-formatted with header lines starting with '>'")
                seq_w_gap_len += len(seq_line)
                seq_len += len(seq_line) - seq_line.count("-")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    if has_sequence:
        seq_lens_w_gap.append(seq_w_gap_len)
        seq_lens.append(seq_len)

    if len(seq_lens_w_gap) == 0:
        out["num_site"] = 0
        out["num_seq"] = 0
        out["len_max"] = 0
        out["len_min"] = 0
        return out

    if len(set(seq_lens_w_gap)) != 1:
        raise ValueError("all FASTA sequences must have the same aligned length")

    out["num_site"] = max(seq_lens_w_gap)
    out["num_seq"] = len(seq_lens)
    out["len_max"] = max(seq_lens)
    out["len_min"] = min(seq_lens)
    return out


def get_iqtree_model_stats(file: Any) -> dict[str, str]:
    """Parse best AIC, AICc, and BIC model names from a gzipped IQ-TREE log."""
    out = {}
    file = coerce_path_argument(file, "file")
    try:
        with gzip.open(file, "rb") as f:
            for line in f:
                decoded = line.decode()
                if "best_model_AIC:" in decoded:
                    out["iqtree_best_AIC"] = decoded.replace("best_model_AIC: ", "").replace("\n", "")
                if "best_model_AICc:" in decoded:
                    out["iqtree_best_AICc"] = decoded.replace("best_model_AICc: ", "").replace("\n", "")
                if "best_model_BIC:" in decoded:
                    out["iqtree_best_BIC"] = decoded.replace("best_model_BIC: ", "").replace("\n", "")
    except UnicodeDecodeError as exc:
        raise ValueError(f"gzip file must contain UTF-8 text: {file}") from exc
    except OSError as exc:
        raise ValueError(f"file is not a readable gzip file: {file}") from exc
    return out


def _regime_parameter_rows(df):
    param_rows = df.loc[df["regime"].isnull(), :]
    invalid = param_rows["param"].isna() | (param_rows["param"].astype(str).str.strip() == "")
    if invalid.any():
        raise ValueError("regime2tree requires non-empty param names for rows with missing regime IDs")
    traits = [column for column in df.columns if column not in {"node_name", "param", "regime"}]
    if not traits:
        raise ValueError("regime2tree requires at least one trait column after node_name/param/regime")
    for param, param_df in param_rows.groupby("param", dropna=False):
        if param_df.loc[:, traits].drop_duplicates().shape[0] > 1:
            raise ValueError(f"regime2tree contains conflicting values for param '{param}' in trait columns")
    return param_rows, traits


def _add_regime_parameters(out, param_rows, traits):
    dedup = param_rows.drop_duplicates(subset="param", keep="first")
    for row in dedup.loc[:, ["param"] + traits].to_numpy():
        out.update({f"{row[0]}_{trait}": value for trait, value in zip(traits, row[1:], strict=True)})


def _add_regime_gamma_values(out, traits):
    for trait in traits:
        alpha_key, sigma_key = f"alpha_{trait}", f"sigma2_{trait}"
        if (alpha_key not in out) or (sigma_key not in out):
            continue
        try:
            alpha_value = float(str(out[alpha_key]))
            sigma_value = float(str(out[sigma_key]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"alpha/sigma2 values must be numeric to compute gamma for trait '{trait}'") from exc
        if (not np.isfinite(alpha_value)) or (not np.isfinite(sigma_value)):
            raise ValueError(f"alpha/sigma2 values must be finite to compute gamma for trait '{trait}'")
        if alpha_value == 0:
            raise ValueError(f"alpha_{trait} must be non-zero to compute gamma_{trait}")
        out[f"gamma_{trait}"] = sigma_value / (2 * alpha_value)


def regime2tree(file: Any) -> dict[str, Any]:
    """Summarize a tab-separated OU regime parameter table."""
    file = coerce_path_argument(file, "file")
    try:
        df = pd.read_csv(file, sep="\t", header=0, index_col=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Failed to read file as UTF-8 tab-separated text: {file}") from exc
    if df.shape[0] == 0:
        raise ValueError("regime2tree requires at least one data row")
    required_columns = {"param", "regime"}
    missing_columns = sorted(required_columns - set(df.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"regime2tree requires columns: {missing_columns}")
    df = df.copy()
    df["regime"] = _validated_regime_series(df, "")
    out: dict[str, object] = {}
    non_nan_regimes = df["regime"].dropna()
    out["num_regime"] = 0 if non_nan_regimes.empty else int(non_nan_regimes.max() + 1)
    param_rows, traits = _regime_parameter_rows(df)
    _add_regime_parameters(out, param_rows, traits)
    if {"alpha", "sigma2"} <= set(param_rows["param"]):
        _add_regime_gamma_values(out, traits)
    return out


def get_dating_method(file: Any) -> str:
    """Read a dating-method file as one newline-free string."""
    file = coerce_path_argument(file, "file")
    try:
        with open(file) as f:
            return f.read().replace("\n", "")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc


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
    b_og = b.loc[b[og_col] == og, ["branch_id", "parent", target_col, return_col]]
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

    def find(self, nl: Any, og: Any, target_value: Any) -> Any:
        """Return the nearest matching ancestor without rebuilding table indexes."""
        _validate_hashable_scalar(nl, "nl must be a hashable scalar branch_id value")
        _validate_hashable_scalar(og, "og must be a hashable value comparable to the orthogroup column")
        b_og = self.tables.get(og)
        if b_og is None or (nl not in b_og.index):
            return np.nan
        return _walk_most_recent(b_og, nl, self.target_col, target_value, self.return_col)


def prepare_most_recent_lookup(
    b: Any,
    target_col: str,
    return_col: str,
    og_col: str = "orthogroup",
) -> MostRecentLookup:
    """Prepare indexes once for repeated :func:`get_most_recent` operations."""
    if not hasattr(b, "columns"):
        raise ValueError("prepare_most_recent_lookup requires a dataframe-like input with columns")
    _validate_column_name(target_col, "target_col")
    _validate_column_name(return_col, "return_col")
    _validate_column_name(og_col, "og_col")
    required_columns = {"branch_id", "parent", target_col, return_col, og_col}
    missing_columns = sorted(required_columns - set(b.columns))
    if missing_columns:
        raise ValueError(f"prepare_most_recent_lookup requires columns: {missing_columns}")

    columns = ["branch_id", "parent", target_col, return_col, og_col]
    prepared_source = b.loc[:, columns]
    _validate_non_missing_series_values(prepared_source["branch_id"], "prepare_most_recent_lookup branch_id column")
    _validate_hashable_series_values(prepared_source["branch_id"], "prepare_most_recent_lookup branch_id column")
    _validate_hashable_series_values(prepared_source["parent"], "prepare_most_recent_lookup parent column")
    _validate_hashable_series_values(prepared_source[og_col], "prepare_most_recent_lookup orthogroup column")

    tables = {}
    for og_value, group in prepared_source.dropna(subset=[og_col]).groupby(og_col, sort=False):
        tables[og_value] = (
            group.loc[:, ["branch_id", "parent", target_col, return_col]]
            .drop_duplicates(subset="branch_id", keep="first")
            .set_index("branch_id", drop=False)
        )
    return MostRecentLookup(target_col, return_col, og_col, tables)


def get_most_recent(
    b: Any,
    nl: Any,
    og: Any,
    target_col: str,
    target_value: Any,
    return_col: str,
    og_col: str = "orthogroup",
) -> Any:
    """Return the nearest node value on the nl->root path matching a target state.

    If the path cannot be followed safely (missing nodes, missing parent, or cycles),
    this function returns np.nan.
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
