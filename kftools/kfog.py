import copy
import gzip
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    from .kfexpression import calc_complementarity, calc_tau
    from .kfphylo import add_numerical_node_labels, check_ultrametric, load_phylo_tree, taxonomic_annotation
except ImportError:
    from kfexpression import calc_complementarity, calc_tau
    from kfphylo import add_numerical_node_labels, check_ultrametric, load_phylo_tree, taxonomic_annotation

NOTUNG_OPT_ROOT_RE = re.compile(
    r"Number of optimal roots:\s*([0-9][0-9,]*)\s*out of\s*([0-9][0-9,]*)",
    flags=re.IGNORECASE,
)
FLOAT_TOKEN_RE = r"[-+]?(?:[0-9]+(?:[.,][0-9]*)?|[.,][0-9]+)(?:[eE][-+]?[0-9]+)?"
NOTUNG_BEST_SCORE_RE = re.compile(
    r"Best rooting score:\s*(.*?)\s*,\s*worst rooting score:\s*(.*?)\s*$",
    flags=re.IGNORECASE,
)
ROOT_RETURNING_RE = re.compile(r"Returning the (.*) tree", flags=re.IGNORECASE)
ROOT_POSITIONS_PREFIX = "root positions with rho peak: "
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


def _validate_boolean_flag(flag_value, argument_name):
    if not isinstance(flag_value, (bool, np.bool_)):
        raise ValueError(f"{argument_name} must be a boolean value")
    return bool(flag_value)


def _validate_hashable_series_values(series, argument_name):
    non_missing_values = series.dropna().to_list()
    unhashable_examples = []
    for value in non_missing_values:
        try:
            hash(value)
        except TypeError:
            unhashable_examples.append(str(value))
            if len(unhashable_examples) >= 5:
                break
    if len(unhashable_examples) > 0:
        raise ValueError(
            f"{argument_name} must contain hashable values; invalid examples: {unhashable_examples}"
        )


def _validate_non_missing_series_values(series, argument_name):
    missing_mask = series.isna()
    if missing_mask.any():
        raise ValueError(f"{argument_name} must not contain missing values")


def _coerce_path_argument(path_value, argument_name='file'):
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


def _parse_int_suffix(line, prefix):
    if (line is None) or (prefix not in line):
        return None
    value = line.replace(prefix, '').strip()
    value = value.replace(',', '')
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float_locale(value):
    text = str(value).strip().replace(' ', '')
    if text == '':
        raise ValueError("empty float token")
    try:
        return float(text)
    except ValueError:
        pass

    exponent_part = ''
    mantissa = text
    if ('e' in text) or ('E' in text):
        split_match = re.match(r"^(.*?)([eE][-+]?[0-9]+)$", text)
        if split_match is None:
            raise ValueError(f"invalid float token: {value}")
        mantissa = split_match.group(1)
        exponent_part = split_match.group(2)

    if (',' in mantissa) and ('.' in mantissa):
        if mantissa.rfind('.') > mantissa.rfind(','):
            # 1,234.56 -> dot decimal, comma thousands
            mantissa = mantissa.replace(',', '')
        else:
            # 1.234,56 -> comma decimal, dot thousands
            mantissa = mantissa.replace('.', '').replace(',', '.')
    elif ',' in mantissa:
        comma_parts = mantissa.split(',')
        if (
            (len(comma_parts) > 2)
            and all(len(part) == 3 for part in comma_parts[1:])
            and (comma_parts[0] not in ('', '+', '-'))
        ):
            # 1,234,567 -> comma thousands (multi-separator form)
            mantissa = ''.join(comma_parts)
        else:
            # 1234,56 -> comma decimal
            mantissa = mantissa.replace(',', '.')
    elif '.' in mantissa:
        dot_parts = mantissa.split('.')
        if (
            (len(dot_parts) > 2)
            and all(len(part) == 3 for part in dot_parts[1:])
            and (dot_parts[0] not in ('', '+', '-'))
        ):
            # 1.234.567 -> dot thousands (multi-separator form)
            mantissa = ''.join(dot_parts)

    return float(mantissa + exponent_part)


def nwk2table(tree, attr='', age=False, parent=False, sister=False):
    age = _validate_boolean_flag(age, "age")
    parent = _validate_boolean_flag(parent, "parent")
    sister = _validate_boolean_flag(sister, "sister")
    if not isinstance(attr, str):
        raise ValueError("attr must be a string")
    if age and (attr != 'dist'):
        raise ValueError("age=True is supported only when attr='dist'")
    tree_format = 0 if attr == 'support' else 1
    if not hasattr(tree, "traverse"):
        try:
            tree = load_phylo_tree(tree, parser=tree_format)
        except TypeError as exc:
            raise ValueError("tree must be a Newick string, path, or tree object") from exc
    tree = add_numerical_node_labels(tree)
    nodes = list(tree.traverse())
    n_nodes = len(nodes)
    age_values = np.empty(n_nodes, dtype=float) if ((attr == 'dist') and age) else None
    if age_values is not None:
        if not check_ultrametric(tree):
            raise ValueError("Tree must be ultrametric when age=True and attr='dist'")
        for node in tree.traverse(strategy="postorder"):
            label = node.branch_id
            if node.is_leaf:
                age_values[label] = 0.0
            else:
                first_child = node.children[0]
                age_values[label] = age_values[first_child.branch_id] + first_child.dist

    children = tree.children
    has_typed_attr = (len(children) > 0) and hasattr(children[0], attr)
    has_attr_for_all_nodes = all(hasattr(node, attr) for node in nodes)
    attr_values_raw = [np.nan] * n_nodes
    for node in nodes:
        label = node.branch_id
        if has_attr_for_all_nodes:
            attr_values_raw[label] = getattr(node, attr)
        else:
            attr_values_raw[label] = getattr(node, attr) if hasattr(node, attr) else np.nan
    if has_typed_attr and has_attr_for_all_nodes:
        sample_attr = getattr(children[0], attr)
        if isinstance(sample_attr, (str, bytes, bytearray)):
            # Preserve variable-length strings instead of truncating to fixed-width dtypes.
            attr_values = np.asarray(attr_values_raw, dtype=object)
        else:
            try:
                attr_values = np.asarray(attr_values_raw, dtype=type(sample_attr))
            except (TypeError, ValueError, OverflowError):
                attr_values = np.asarray(attr_values_raw, dtype=object)
    else:
        attr_values = np.asarray(attr_values_raw, dtype=object)
    parent_values = np.full(n_nodes, -1, dtype=np.int64) if parent else None
    sister_values = np.full(n_nodes, -1, dtype=np.int64) if sister else None

    for node in nodes:
        label = node.branch_id
        if parent_values is not None and (not node.is_root):
            parent_values[label] = node.up.branch_id
        if sister_values is not None and (not node.is_root):
            siblings = node.up.children
            if len(siblings) == 2:
                sister_node = siblings[1] if siblings[0] is node else siblings[0]
                sister_values[label] = sister_node.branch_id
            else:
                sister_nodes = node.get_sisters()
                if len(sister_nodes) > 0:
                    sister_values[label] = sister_nodes[0].branch_id

    data = {'branch_id': np.arange(n_nodes, dtype=np.int64)}
    data[attr] = attr_values
    if age_values is not None:
        data['age'] = age_values
    if parent_values is not None:
        data['parent'] = parent_values
    if sister_values is not None:
        data['sister'] = sister_values
    df = pd.DataFrame(data)
    return df

def node_gene2species(gene_tree, species_tree, is_ultrametric=False):
    is_ultrametric = _validate_boolean_flag(is_ultrametric, "is_ultrametric")
    if not hasattr(gene_tree, "traverse"):
        try:
            gene_tree = load_phylo_tree(gene_tree, parser=1)
        except TypeError as exc:
            raise ValueError("gene_tree must be a Newick string, path, or tree object") from exc
    if not hasattr(species_tree, "traverse"):
        try:
            species_tree = load_phylo_tree(species_tree, parser=1)
        except TypeError as exc:
            raise ValueError("species_tree must be a Newick string, path, or tree object") from exc
    species_leaf_names = list(species_tree.leaf_names())
    invalid_species_leaf_names = [
        species_leaf_name
        for species_leaf_name in species_leaf_names
        if (not isinstance(species_leaf_name, str)) or (species_leaf_name.strip() == "")
    ]
    if len(invalid_species_leaf_names) > 0:
        raise ValueError(
            "species_tree leaf names must be non-empty strings for node_gene2species"
        )
    species_name_counts = {}
    for species_leaf_name in species_leaf_names:
        species_name_counts[species_leaf_name] = species_name_counts.get(species_leaf_name, 0) + 1
    duplicate_species_names = sorted(
        [species_leaf_name for species_leaf_name, species_count in species_name_counts.items() if species_count > 1]
    )
    if len(duplicate_species_names) > 0:
        raise ValueError(
            "species_tree leaf names must be unique for node_gene2species; "
            f"duplicates: {duplicate_species_names}"
        )
    gene_tree2 = copy.deepcopy(gene_tree)
    gene_tree2 = add_numerical_node_labels(gene_tree2)
    if is_ultrametric and (not check_ultrametric(gene_tree2)):
        raise ValueError("gene_tree must be ultrametric when is_ultrametric=True")
    if is_ultrametric:
        try:
            species_is_ultrametric = check_ultrametric(species_tree)
        except ValueError as exc:
            raise ValueError(
                "species_tree must be ultrametric with finite non-negative branch lengths "
                "when is_ultrametric=True"
            ) from exc
        if not species_is_ultrametric:
            raise ValueError("species_tree must be ultrametric when is_ultrametric=True")
    for leaf in gene_tree2.leaves():
        leaf_name_split = leaf.name.split("_", 2)
        if len(leaf_name_split) < 2:
            raise ValueError(f"Gene leaf name must contain species information with '_': {leaf.name}")
        binom_name = leaf_name_split[0] + "_" + leaf_name_split[1]
        leaf.name = binom_name
    tip_set_diff = set(gene_tree2.leaf_names()) - set(species_tree.leaf_names())
    if tip_set_diff:
        sys.stderr.write(f"Warning. A total of {len(tip_set_diff)} species are missing in the species tree: {str(tip_set_diff)}\n")
    if is_ultrametric:
        cn = ["branch_id", "spnode_coverage", 'spnode_age']
    else:
        cn = ["branch_id", "spnode_coverage"]
    species_nodes = list(species_tree.traverse(strategy="postorder"))
    species_names = {sn: (sn.name or '').replace('\'', '') for sn in species_nodes}
    species_leaf_node = {leaf.name: leaf for leaf in species_tree.leaves()}
    species_depth = {}
    for sn in species_tree.traverse(strategy="preorder"):
        species_depth[sn] = 0 if sn.is_root else (species_depth[sn.up] + 1)
    if is_ultrametric:
        species_age = {}
        for sn in species_nodes:
            if sn.is_leaf:
                species_age[sn] = 0.0
            else:
                first_child = sn.children[0]
                species_age[sn] = species_age[first_child] + first_child.dist
        species_up_age = {}
        for sn in species_nodes:
            if sn.is_root:
                species_up_age[sn] = np.inf
            else:
                species_up_age[sn] = species_age[sn.up]

    lca_cache = {}

    def pair_lca(node_a, node_b):
        if node_a is node_b:
            return node_a
        key = (node_a, node_b) if id(node_a) <= id(node_b) else (node_b, node_a)
        cached = lca_cache.get(key)
        if cached is not None:
            return cached
        a = node_a
        b = node_b
        depth_a = species_depth[a]
        depth_b = species_depth[b]
        while depth_a > depth_b:
            a = a.up
            depth_a -= 1
        while depth_b > depth_a:
            b = b.up
            depth_b -= 1
        while a is not b:
            a = a.up
            b = b.up
        lca_cache[key] = a
        return a

    gene_nodes = list(gene_tree2.traverse(strategy="postorder"))
    gene_coverage = {}
    gene_has_missing_species = {}
    if is_ultrametric:
        gene_age = {}
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
            covered_species_node = pair_lca(covered_species_node, gene_coverage[child])
        gene_coverage[gn] = covered_species_node

    rows = []
    for gn in gene_nodes:
        coverage_node = gene_coverage[gn]
        row = {
            "branch_id": gn.branch_id,
            "spnode_coverage": "" if coverage_node is None else species_names[coverage_node],
        }
        if is_ultrametric:
            row["spnode_age"] = ""
            if coverage_node is not None:
                gn_age = gene_age[gn]
                current_species_node = coverage_node
                while current_species_node is not None:
                    if (gn_age >= species_age[current_species_node]) and (gn_age < species_up_age[current_species_node]):
                        row["spnode_age"] = species_names[current_species_node]
                        break
                    current_species_node = current_species_node.up
        rows.append(row)
    return pd.DataFrame(rows, columns=cn)

def ou2table(regime_file, leaf_file, input_tree_file):
    regime_file = _coerce_path_argument(regime_file, 'regime_file')
    leaf_file = _coerce_path_argument(leaf_file, 'leaf_file')
    input_tree_file = _coerce_path_argument(input_tree_file, 'input_tree_file')
    if (not os.path.exists(input_tree_file)) or (not os.path.isfile(input_tree_file)):
        raise ValueError(f"input_tree_file must be an existing file path: {input_tree_file}")
    try:
        df_regime = pd.read_csv(regime_file, sep="\t")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Failed to read regime_file as UTF-8 tab-separated text: {regime_file}") from exc
    try:
        df_leaf = pd.read_csv(leaf_file, sep="\t")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Failed to read leaf_file as UTF-8 tab-separated text: {leaf_file}") from exc
    required_regime_columns = {"node_name", "regime"}
    missing_regime_columns = sorted(required_regime_columns - set(df_regime.columns))
    if len(missing_regime_columns) > 0:
        raise ValueError(f"regime_file requires columns: {missing_regime_columns}")
    required_leaf_columns = ("node_name", "param", "regime")
    required_leaf_columns_set = set(required_leaf_columns)
    missing_leaf_columns = sorted(required_leaf_columns_set - set(df_leaf.columns))
    if len(missing_leaf_columns) > 0:
        raise ValueError(f"leaf_file requires columns: {missing_leaf_columns}")
    if df_leaf.shape[1] <= 3:
        raise ValueError("leaf_file must include at least one trait column after node_name/param/regime")
    df_regime = df_regime.copy()
    df_leaf = df_leaf.copy()
    for df_name, df_obj in [("regime_file", df_regime), ("leaf_file", df_leaf)]:
        regime_numeric = pd.to_numeric(df_obj["regime"], errors="coerce")
        invalid_regime_mask = df_obj["regime"].notna() & regime_numeric.isna()
        if invalid_regime_mask.any():
            invalid_values = sorted(set(df_obj.loc[invalid_regime_mask, "regime"].astype(str)))
            raise ValueError(f"{df_name} regime column must be numeric or NaN; invalid values: {invalid_values}")
        non_finite_mask = regime_numeric.notna() & (~np.isfinite(regime_numeric.to_numpy(dtype=float, copy=False)))
        if non_finite_mask.any():
            invalid_values = sorted(set(df_obj.loc[non_finite_mask, "regime"].astype(str)))
            raise ValueError(f"{df_name} regime column must contain finite numeric values; invalid values: {invalid_values}")
        non_integer_mask = regime_numeric.notna() & (regime_numeric != np.floor(regime_numeric))
        if non_integer_mask.any():
            invalid_values = sorted(set(df_obj.loc[non_integer_mask, "regime"].astype(str)))
            raise ValueError(f"{df_name} regime column must contain integer IDs; invalid values: {invalid_values}")
        negative_mask = regime_numeric.notna() & (regime_numeric < 0)
        if negative_mask.any():
            invalid_values = sorted(set(df_obj.loc[negative_mask, "regime"].astype(str)))
            raise ValueError(f"{df_name} regime column must contain non-negative IDs; invalid values: {invalid_values}")
        too_large_mask = regime_numeric.notna() & (regime_numeric > INT64_MAX)
        if too_large_mask.any():
            invalid_values = sorted(set(df_obj.loc[too_large_mask, "regime"].astype(str)))
            raise ValueError(
                f"{df_name} regime column must be <= {INT64_MAX} to avoid integer overflow; "
                f"invalid values: {invalid_values}"
            )
        df_obj["regime"] = regime_numeric
    trait_columns = [column_name for column_name in df_leaf.columns if column_name not in required_leaf_columns_set]
    if len(trait_columns) == 0:
        raise ValueError("leaf_file must include at least one trait column after node_name/param/regime")
    for trait_col in trait_columns:
        trait_numeric = pd.to_numeric(df_leaf[trait_col], errors="coerce")
        invalid_trait_mask = df_leaf[trait_col].notna() & trait_numeric.isna()
        if invalid_trait_mask.any():
            invalid_values = sorted(set(df_leaf.loc[invalid_trait_mask, trait_col].astype(str)))
            raise ValueError(
                f"leaf_file trait column '{trait_col}' must be numeric or NaN; invalid values: {invalid_values}"
            )
        df_leaf[trait_col] = trait_numeric
    tree = load_phylo_tree(input_tree_file, parser=1)
    tree = add_numerical_node_labels(tree)
    nodes = list(tree.traverse())
    num_nodes = len(nodes)
    tissues = trait_columns
    if 'expectations' in df_leaf['param'].values:
        df_leaf.loc[(df_leaf['param'] == 'expectations'), 'param'] = 'mu'
    cn1 = ["branch_id", "regime", "is_shift", "num_child_shift"]
    cn2 = ["tau", "delta_tau", "delta_maxmu", "mu_complementarity"]
    cn3 = ["mu_" + tissue for tissue in tissues]
    cn = cn1 + cn2 + cn3
    regime_map = {}
    regime_rows_non_nan = df_regime.loc[df_regime["regime"].notna(), ["node_name", "regime"]].copy()
    invalid_node_name_mask = regime_rows_non_nan["node_name"].map(
        lambda node_name: (not isinstance(node_name, str)) or (node_name.strip() == "")
    )
    if invalid_node_name_mask.any():
        invalid_values = sorted(set(regime_rows_non_nan.loc[invalid_node_name_mask, "node_name"].astype(str)))
        raise ValueError(
            "regime_file node_name column must contain non-empty string values when regime is provided; "
            f"invalid values: {invalid_values}"
        )
    if regime_rows_non_nan.shape[0] > 0:
        regime_nunique = regime_rows_non_nan.groupby("node_name")["regime"].nunique(dropna=True)
        conflicting_node_names = sorted(regime_nunique.index[regime_nunique > 1].tolist())
        if len(conflicting_node_names) > 0:
            raise ValueError(
                "regime_file contains conflicting regime IDs for node_name values: "
                f"{conflicting_node_names}"
            )
    named_node_names = [
        node.name
        for node in nodes
        if isinstance(node.name, str) and (node.name.strip() != "")
    ]
    tree_node_name_counts = {}
    for node_name in named_node_names:
        tree_node_name_counts[node_name] = tree_node_name_counts.get(node_name, 0) + 1
    duplicate_tree_node_names = sorted(
        [node_name for node_name, count in tree_node_name_counts.items() if count > 1]
    )
    if len(duplicate_tree_node_names) > 0:
        raise ValueError(
            "input_tree_file contains duplicate non-empty node names that make regime mapping ambiguous: "
            f"{duplicate_tree_node_names}"
        )
    known_node_names = set(named_node_names)
    unknown_node_names = sorted(
        {
            node_name
            for node_name in regime_rows_non_nan["node_name"].tolist()
            if node_name not in known_node_names
        }
    )
    if len(unknown_node_names) > 0:
        raise ValueError(
            "regime_file contains node_name values not present in input_tree_file: "
            f"{unknown_node_names}"
        )
    for node_name, regime in regime_rows_non_nan.itertuples(index=False, name=None):
        if node_name not in regime_map:
            regime_map[node_name] = int(regime)
    for node in tree.traverse(strategy="preorder"):
        if node.is_root:
            node.regime = regime_map.get(node.name, 0)
        else:
            node.regime = regime_map.get(node.name, node.up.regime)
    is_mu = (df_leaf['param'] == 'mu')
    is_cols = [c not in ['node_name', 'param'] for c in df_leaf.columns]
    df_leaf_unique = df_leaf.loc[is_mu, df_leaf.columns[is_cols]]
    df_leaf_unique = df_leaf_unique.drop_duplicates()
    df_leaf_unique = df_leaf_unique.groupby(by='regime').mean()
    mu_table = df_leaf_unique.loc[:, tissues]
    mu_by_regime = {
        regime: values
        for regime, values in zip(mu_table.index.to_numpy(), mu_table.to_numpy())
    }
    observed_regimes = {node.regime for node in nodes}
    missing_regimes = sorted(regime for regime in observed_regimes if regime not in mu_by_regime)
    if missing_regimes:
        raise ValueError(f"Missing mu values for regime IDs: {missing_regimes}")

    branch_id = np.empty(num_nodes, dtype=np.int64)
    regime = np.empty(num_nodes, dtype=np.int64)
    is_shift = np.empty(num_nodes, dtype=np.int64)
    num_child_shift = np.full(num_nodes, np.nan, dtype=float)
    mu_values = np.empty((num_nodes, len(cn3)), dtype=float)
    parent_labels = np.empty(num_nodes, dtype=np.int64)
    shift_pairs = []
    for row_idx, node in enumerate(nodes):
        node_label = node.branch_id
        shift_flag = int((not node.is_root) and (node.regime != node.up.regime))
        branch_id[row_idx] = node_label
        regime[row_idx] = node.regime
        is_shift[row_idx] = shift_flag
        mu_values[row_idx, :] = mu_by_regime[node.regime]
        parent_labels[row_idx] = -1 if node.is_root else node.up.branch_id
        if not node.is_leaf:
            children = node.get_children()
            num_child_shift[row_idx] = sum(int(node.regime != child.regime) for child in children)
        if shift_flag:
            sisters = node.get_sisters()
            if len(sisters) > 0:
                shift_pairs.append((node_label, sisters[0].branch_id))

    df = pd.DataFrame(
        {
            "branch_id": branch_id,
            "regime": regime,
            "is_shift": is_shift,
            "num_child_shift": num_child_shift,
        }
    )
    for col_idx, col in enumerate(cn3):
        df[col] = mu_values[:, col_idx]
    tau_values = calc_tau(df, cn3, unlog2=True, unPlus1=True)
    df["tau"] = tau_values
    tau_by_label = np.empty(num_nodes, dtype=float)
    tau_by_label[branch_id] = tau_values
    parent_labels_safe = parent_labels.copy()
    parent_labels_safe[parent_labels_safe == -1] = 0
    delta_tau = tau_values - tau_by_label[parent_labels_safe]
    delta_tau[parent_labels == -1] = np.nan
    df["delta_tau"] = delta_tau

    mu_max_values = mu_values.max(axis=1)
    mu_unlog_values = np.clip(np.exp2(mu_values) - 1, a_min=0, a_max=None)
    label_to_idx = np.empty(num_nodes, dtype=np.int64)
    label_to_idx[branch_id] = np.arange(num_nodes, dtype=np.int64)
    delta_maxmu = np.full(num_nodes, np.nan, dtype=float)
    mu_complementarity = np.full(num_nodes, np.nan, dtype=float)
    for my_label, sis_label in shift_pairs:
        my_idx = label_to_idx[my_label]
        sis_idx = label_to_idx[sis_label]
        delta_maxmu[my_idx] = float(mu_max_values[my_idx] - mu_max_values[sis_idx])
        mu_complementarity[my_idx] = calc_complementarity(mu_unlog_values[my_idx], mu_unlog_values[sis_idx])
    df["delta_maxmu"] = delta_maxmu
    df["mu_complementarity"] = mu_complementarity
    return df.loc[:, cn]

def get_misc_node_statistics(tree_file, tax_annot=False):
    tax_annot = _validate_boolean_flag(tax_annot, "tax_annot")
    try:
        tree = load_phylo_tree(tree_file, parser=1)
    except TypeError as exc:
        raise ValueError("tree_file must be a Newick string, path, or tree object") from exc
    tree = add_numerical_node_labels(tree)
    cn1 = ["branch_id", "taxon", "taxid", "num_sp", "num_leaf", "so_event", "dup_conf_score"]
    cn2 = ["parent", "sister", "child1", "child2", "so_event_parent"]
    cn = cn1 + cn2
    nodes = list(tree.traverse())
    if tax_annot:
        tree = taxonomic_annotation(tree)
    else:
        for node in nodes:
            node.taxid = -999
            if node.is_leaf:
                name_split = node.name.split('_', 2)
                if len(name_split) >= 2:
                    node.sci_name = name_split[0] + ' ' + name_split[1]
                else:
                    node.sci_name = node.name
            else:
                node.sci_name = ''

    species_index = {}
    species_mask_by_node = {}
    num_leaf_by_node = {}
    dup_conf_score_by_node = {}
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf:
            species_id = species_index.setdefault(node.sci_name, len(species_index))
            species_mask_by_node[node] = (1 << species_id)
            num_leaf_by_node[node] = 1
            dup_conf_score_by_node[node] = 0.0
            continue
        children = node.children
        species_mask = 0
        num_leaf = 0
        for child in children:
            species_mask |= species_mask_by_node[child]
            num_leaf += num_leaf_by_node[child]
        species_mask_by_node[node] = species_mask
        num_leaf_by_node[node] = num_leaf
        if len(children) >= 2:
            species_seen_once = 0
            species_seen_multiple = 0
            for child in children:
                child_mask = species_mask_by_node[child]
                species_seen_multiple |= (species_seen_once & child_mask)
                species_seen_once |= child_mask
            union_count = species_seen_once.bit_count()
            if union_count == 0:
                dup_conf_score_by_node[node] = 0.0
            else:
                dup_conf_score_by_node[node] = species_seen_multiple.bit_count() / union_count
        else:
            dup_conf_score_by_node[node] = 0.0

    n_nodes = len(nodes)
    branch_id = np.empty(n_nodes, dtype=np.int64)
    taxon = np.empty(n_nodes, dtype=object)
    taxid = np.empty(n_nodes, dtype=np.int64)
    num_sp = np.empty(n_nodes, dtype=np.int64)
    num_leaf = np.empty(n_nodes, dtype=np.int64)
    so_event = np.full(n_nodes, "L", dtype=object)
    dup_conf_score = np.zeros(n_nodes, dtype=float)
    parent = np.full(n_nodes, -999, dtype=np.int64)
    sister = np.full(n_nodes, -999, dtype=np.int64)
    child1 = np.full(n_nodes, -999, dtype=np.int64)
    child2 = np.full(n_nodes, -999, dtype=np.int64)
    so_event_parent = np.full(n_nodes, "S", dtype=object)

    for row_idx, node in enumerate(nodes):
        label = node.branch_id
        node_dup_conf_score = dup_conf_score_by_node.get(node, 0.0)
        branch_id[row_idx] = label
        taxon[row_idx] = str(node.sci_name)
        taxid[row_idx] = node.taxid
        num_sp[row_idx] = species_mask_by_node[node].bit_count()
        num_leaf[row_idx] = num_leaf_by_node[node]
        if hasattr(node.up, "branch_id"):
            parent[row_idx] = node.up.branch_id
        if node.up is not None:
            siblings = node.up.children
            if len(siblings) == 2:
                sister_node = siblings[1] if siblings[0] is node else siblings[0]
                sister[row_idx] = sister_node.branch_id
            else:
                sister_nodes = node.get_sisters()
                if len(sister_nodes) > 0:
                    sister[row_idx] = sister_nodes[0].branch_id
        if not node.is_leaf:
            if len(node.children) >= 1:
                child1[row_idx] = node.children[0].branch_id
            if len(node.children) >= 2:
                child2[row_idx] = node.children[1].branch_id
            dup_conf_score[row_idx] = node_dup_conf_score
            so_event[row_idx] = "D" if node_dup_conf_score > 0 else "S"
        if (node.up is not None) and (dup_conf_score_by_node.get(node.up, 0.0) > 0):
            so_event_parent[row_idx] = "D"

    return pd.DataFrame(
        {
            "branch_id": branch_id,
            "taxon": taxon,
            "taxid": taxid,
            "num_sp": num_sp,
            "num_leaf": num_leaf,
            "so_event": so_event,
            "dup_conf_score": dup_conf_score,
            "parent": parent,
            "sister": sister,
            "child1": child1,
            "child2": child2,
            "so_event_parent": so_event_parent,
        },
        columns=cn,
    )

def compute_delta(df, column):
    if not hasattr(df, 'columns'):
        raise ValueError("compute_delta requires a dataframe-like input with columns")
    _validate_column_name(column, "column")
    required_columns = {'branch_id', 'parent', column}
    missing_columns = sorted(required_columns - set(df.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"compute_delta requires columns: {missing_columns}")
    out = df.copy()
    _validate_non_missing_series_values(out['branch_id'], "compute_delta branch_id column")
    _validate_hashable_series_values(out['branch_id'], "compute_delta branch_id column")
    _validate_hashable_series_values(out['parent'], "compute_delta parent column")
    if not out['branch_id'].is_unique:
        raise ValueError("compute_delta requires unique branch_id values")
    numeric_column = pd.to_numeric(out[column], errors='coerce')
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
    parent_column = f'parent_{column}'
    value_by_label = out.set_index('branch_id')[column]
    out[parent_column] = out['parent'].map(value_by_label)
    out[f'delta_{column}'] = out[column] - out[parent_column]
    out = out.drop(parent_column, axis=1)
    return out

def get_notung_root_stats(file):
    file = _coerce_path_argument(file, 'file')
    out = {}
    try:
        with open(file) as f:
            for line in f:
                m_opt = NOTUNG_OPT_ROOT_RE.search(line)
                if m_opt is not None:
                    out['ntg_num_opt_root'] = int(m_opt.group(1).replace(',', ''))
                m_best = NOTUNG_BEST_SCORE_RE.search(line)
                if m_best is not None:
                    try:
                        best_score = _parse_float_locale(m_best.group(1))
                        worst_score = _parse_float_locale(m_best.group(2))
                    except ValueError:
                        continue
                    out['ntg_best_root_score'] = best_score
                    out['ntg_worst_root_score'] = worst_score
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    return out

def get_notung_reconcil_stats(file):
    file = _coerce_path_argument(file, 'file')
    out = {}
    count_patterns = [
        ('ntg_num_dup', NOTUNG_DUP_RE),
        ('ntg_num_codiv', NOTUNG_CODIV_RE),
        ('ntg_num_transfer', NOTUNG_TRANSFER_RE),
        ('ntg_num_loss', NOTUNG_LOSS_RE),
        ('ntg_num_polytomy', NOTUNG_POLYTOMY_RE),
    ]
    try:
        with open(file) as f:
            for line in f:
                for key, pattern in count_patterns:
                    m = pattern.search(line)
                    if m is not None:
                        out[key] = int(m.group(1).replace(',', ''))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    return out


def get_root_stats(file):
    file = _coerce_path_argument(file, 'file')
    out = {}
    try:
        with open(file) as f:
            for line in f:
                m_pos = ROOT_POSITIONS_RE.search(line)
                if m_pos is not None:
                    positions = m_pos.group(1).strip()
                    if positions == '':
                        out['num_rho_peak'] = 0
                    else:
                        tokens = [tok for tok in re.split(r'[\s,]+', positions) if tok != '']
                        placeholder_tokens = {'-', 'none', 'na', 'n/a', 'null'}
                        valid_tokens = [tok for tok in tokens if tok.lower() not in placeholder_tokens]
                        out['num_rho_peak'] = len(valid_tokens)
                m = ROOT_RETURNING_RE.search(line)
                if m is not None:
                    rooting_method = m.group(1).strip()
                    if rooting_method.lower().startswith('first '):
                        rooting_method = rooting_method[6:].strip()
                    out['rooting_method'] = rooting_method
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    return out

def get_aln_stats(file):
    file = _coerce_path_argument(file, 'file')
    out = {}
    seq_lens_w_gap = []
    seq_lens = []
    seq_w_gap_len = 0
    seq_len = 0
    has_sequence = False
    try:
        with open(file) as f:
            for line in f:
                if line.startswith('>'):
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
                seq_len += len(seq_line) - seq_line.count('-')
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    if has_sequence:
        seq_lens_w_gap.append(seq_w_gap_len)
        seq_lens.append(seq_len)

    if len(seq_lens_w_gap) == 0:
        out['num_site'] = 0
        out['num_seq'] = 0
        out['len_max'] = 0
        out['len_min'] = 0
        return out

    out['num_site'] = max(seq_lens_w_gap)
    out['num_seq'] = len(seq_lens)
    out['len_max'] = max(seq_lens)
    out['len_min'] = min(seq_lens)
    return out


def get_iqtree_model_stats(file):
    out = {}
    file = _coerce_path_argument(file, 'file')
    try:
        with gzip.open(file, 'rb') as f:
            for line in f:
                decoded = line.decode()
                if 'best_model_AIC:' in decoded:
                    out['iqtree_best_AIC'] = decoded.replace('best_model_AIC: ', '').replace('\n', '')
                if 'best_model_AICc:' in decoded:
                    out['iqtree_best_AICc'] = decoded.replace('best_model_AICc: ', '').replace('\n', '')
                if 'best_model_BIC:' in decoded:
                    out['iqtree_best_BIC'] = decoded.replace('best_model_BIC: ', '').replace('\n', '')
    except UnicodeDecodeError as exc:
        raise ValueError(f"gzip file must contain UTF-8 text: {file}") from exc
    except OSError as exc:
        raise ValueError(f"file is not a readable gzip file: {file}") from exc
    return out



def regime2tree(file):
    file = _coerce_path_argument(file, 'file')
    try:
        df = pd.read_csv(file, sep='\t', header=0, index_col=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Failed to read file as UTF-8 tab-separated text: {file}") from exc
    if df.shape[0] == 0:
        raise ValueError("regime2tree requires at least one data row")
    required_columns = {'param', 'regime'}
    missing_columns = sorted(required_columns - set(df.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"regime2tree requires columns: {missing_columns}")
    regime_numeric = pd.to_numeric(df['regime'], errors='coerce')
    invalid_regime_mask = df['regime'].notna() & regime_numeric.isna()
    if invalid_regime_mask.any():
        invalid_values = sorted(set(df.loc[invalid_regime_mask, 'regime'].astype(str)))
        raise ValueError(f"regime column must be numeric or NaN; invalid values: {invalid_values}")
    non_finite_regime_mask = regime_numeric.notna() & (~np.isfinite(regime_numeric.to_numpy(dtype=float, copy=False)))
    if non_finite_regime_mask.any():
        invalid_values = sorted(set(df.loc[non_finite_regime_mask, 'regime'].astype(str)))
        raise ValueError(f"regime column must contain finite numeric values; invalid values: {invalid_values}")
    non_integer_regime_mask = regime_numeric.notna() & (regime_numeric != np.floor(regime_numeric))
    if non_integer_regime_mask.any():
        invalid_values = sorted(set(df.loc[non_integer_regime_mask, 'regime'].astype(str)))
        raise ValueError(f"regime column must contain integer IDs; invalid values: {invalid_values}")
    negative_regime_mask = regime_numeric.notna() & (regime_numeric < 0)
    if negative_regime_mask.any():
        invalid_values = sorted(set(df.loc[negative_regime_mask, 'regime'].astype(str)))
        raise ValueError(f"regime column must contain non-negative IDs; invalid values: {invalid_values}")
    too_large_regime_mask = regime_numeric.notna() & (regime_numeric > INT64_MAX)
    if too_large_regime_mask.any():
        invalid_values = sorted(set(df.loc[too_large_regime_mask, 'regime'].astype(str)))
        raise ValueError(
            f"regime column must be <= {INT64_MAX} to avoid integer overflow; invalid values: {invalid_values}"
        )
    df = df.copy()
    df['regime'] = regime_numeric
    out = {}
    non_nan_regimes = df['regime'].dropna()
    if non_nan_regimes.shape[0] == 0:
        out['num_regime'] = 0
    else:
        out['num_regime'] = int(non_nan_regimes.max() + 1)
    param_rows = df.loc[df['regime'].isnull(), :]
    invalid_param_mask = param_rows["param"].isna() | (param_rows["param"].astype(str).str.strip() == "")
    if invalid_param_mask.any():
        raise ValueError("regime2tree requires non-empty param names for rows with missing regime IDs")
    params_set = set(param_rows.loc[:, 'param'].values)
    non_trait_columns = {"node_name", "param", "regime"}
    traits = [column_name for column_name in df.columns if column_name not in non_trait_columns]
    if len(traits) == 0:
        raise ValueError("regime2tree requires at least one trait column after node_name/param/regime")
    if param_rows.shape[0] > 0:
        for param, param_df in param_rows.groupby("param", dropna=False):
            if param_df.loc[:, traits].drop_duplicates().shape[0] > 1:
                raise ValueError(
                    f"regime2tree contains conflicting values for param '{param}' in trait columns"
                )
    dedup = param_rows.drop_duplicates(subset='param', keep='first')
    rows = dedup.loc[:, ['param'] + traits].to_numpy()
    for row in rows:
        param = row[0]
        out.update({f'{param}_{trait}': value for trait, value in zip(traits, row[1:])})
    if all(key in params_set for key in ['alpha', 'sigma2']):
        for trait in traits:
            alpha_key = 'alpha_' + trait
            sigma_key = 'sigma2_' + trait
            if (alpha_key not in out) or (sigma_key not in out):
                continue
            try:
                alpha_value = float(out[alpha_key])
                sigma_value = float(out[sigma_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"alpha/sigma2 values must be numeric to compute gamma for trait '{trait}'"
                ) from exc
            if (not np.isfinite(alpha_value)) or (not np.isfinite(sigma_value)):
                raise ValueError(
                    f"alpha/sigma2 values must be finite to compute gamma for trait '{trait}'"
                )
            if alpha_value == 0:
                raise ValueError(
                    f"alpha_{trait} must be non-zero to compute gamma_{trait}"
                )
            out['gamma_' + trait] = sigma_value / (2 * alpha_value)
    return out


def get_dating_method(file):
    file = _coerce_path_argument(file, 'file')
    try:
        with open(file) as f:
            return f.read().replace('\n', '')
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc


def get_most_recent(b, nl, og, target_col, target_value, return_col, og_col='orthogroup'):
    """Return the nearest node value on the nl->root path matching a target state.

    If the path cannot be followed safely (missing nodes, missing parent, or cycles),
    this function returns np.nan.
    """
    if not hasattr(b, 'columns'):
        raise ValueError("get_most_recent requires a dataframe-like input with columns")
    _validate_column_name(target_col, "target_col")
    _validate_column_name(return_col, "return_col")
    _validate_column_name(og_col, "og_col")
    try:
        hash(nl)
    except TypeError as exc:
        raise ValueError("nl must be a hashable scalar branch_id value") from exc
    try:
        hash(og)
    except TypeError as exc:
        raise ValueError("og must be a hashable value comparable to the orthogroup column") from exc
    required_columns = {'branch_id', 'parent', target_col, return_col, og_col}
    missing_columns = sorted(required_columns - set(b.columns))
    if len(missing_columns) > 0:
        raise ValueError(f"get_most_recent requires columns: {missing_columns}")
    b_og = b.loc[b[og_col] == og, ['branch_id', 'parent', target_col, return_col]]
    _validate_non_missing_series_values(b_og['branch_id'], "get_most_recent branch_id column")
    _validate_hashable_series_values(b_og['branch_id'], "get_most_recent branch_id column")
    _validate_hashable_series_values(b_og['parent'], "get_most_recent parent column")
    b_og = b_og.drop_duplicates(subset='branch_id', keep='first').set_index('branch_id', drop=False)
    if b_og.empty or (nl not in b_og.index):
        return np.nan
    current_nl = nl
    visited_nl = set()
    while True:
        if current_nl in visited_nl:
            return np.nan
        if current_nl not in b_og.index:
            return np.nan
        visited_nl.add(current_nl)
        current_value = b_og.at[current_nl, target_col]
        if current_value == target_value:
            return b_og.at[current_nl, return_col]
        current_parent = b_og.at[current_nl, 'parent']
        if pd.isna(current_parent):
            return np.nan
        current_nl = current_parent
