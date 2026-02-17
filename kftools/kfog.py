import copy
import gzip
import re
import sys

import ete3
import numpy
import pandas

try:
    from .kfexpression import calc_complementarity, calc_tau
    from .kfphylo import add_numerical_node_labels, check_ultrametric, taxonomic_annotation
except ImportError:
    from kfexpression import calc_complementarity, calc_tau
    from kfphylo import add_numerical_node_labels, check_ultrametric, taxonomic_annotation

NOTUNG_OPT_ROOT_RE = re.compile(r"Number of optimal roots: ([0-9]+) out of ([0-9]+)")
NOTUNG_BEST_SCORE_RE = re.compile(r"Best rooting score: (\d*[.,]?\d*), worst rooting score: (\d*[.,]?\d*)")
ROOT_RETURNING_RE = re.compile(r"Returning the (.*) tree")
ROOT_POSITIONS_PREFIX = "root positions with rho peak: "


def nwk2table(tree, attr='', age=False, parent=False, sister=False):
    tree_format = 0 if attr == 'support' else 1
    if isinstance(tree, str):
        tree = ete3.PhyloNode(tree, format=tree_format)
    tree = add_numerical_node_labels(tree)
    nodes = list(tree.traverse())
    n_nodes = len(nodes)
    age_values = numpy.empty(n_nodes, dtype=float) if ((attr == 'dist') and age) else None
    if age_values is not None:
        assert check_ultrametric(tree)
        for node in tree.traverse(strategy="postorder"):
            label = node.numerical_label
            if node.is_leaf():
                age_values[label] = 0.0
            else:
                first_child = node.children[0]
                age_values[label] = age_values[first_child.numerical_label] + first_child.dist

    children = tree.children
    has_typed_attr = (len(children) > 0) and hasattr(children[0], attr)
    has_attr_for_all_nodes = all(hasattr(node, attr) for node in nodes)
    if has_typed_attr and has_attr_for_all_nodes:
        attr_dtype = type(getattr(children[0], attr))
        attr_values = numpy.empty(n_nodes, dtype=attr_dtype)
    else:
        attr_values = numpy.empty(n_nodes, dtype=object)
    parent_values = numpy.full(n_nodes, -1, dtype=numpy.int64) if parent else None
    sister_values = numpy.full(n_nodes, -1, dtype=numpy.int64) if sister else None

    for node in nodes:
        label = node.numerical_label
        if has_attr_for_all_nodes:
            attr_values[label] = getattr(node, attr)
        else:
            attr_values[label] = getattr(node, attr) if hasattr(node, attr) else numpy.nan
        if parent_values is not None and (not node.is_root()):
            parent_values[label] = node.up.numerical_label
        if sister_values is not None and (not node.is_root()):
            siblings = node.up.children
            if len(siblings) == 2:
                sister_node = siblings[1] if siblings[0] is node else siblings[0]
                sister_values[label] = sister_node.numerical_label
            else:
                sister_values[label] = node.get_sisters()[0].numerical_label

    data = {'numerical_label': numpy.arange(n_nodes, dtype=numpy.int64)}
    data[attr] = attr_values
    if age_values is not None:
        data['age'] = age_values
    if parent_values is not None:
        data['parent'] = parent_values
    if sister_values is not None:
        data['sister'] = sister_values
    df = pandas.DataFrame(data)
    return df

def node_gene2species(gene_tree, species_tree, is_ultrametric=False):
    gene_tree2 = copy.deepcopy(gene_tree)
    gene_tree2 = add_numerical_node_labels(gene_tree2)
    if is_ultrametric:
        assert check_ultrametric(gene_tree2)
    for leaf in gene_tree2.iter_leaves():
        leaf_name_split = leaf.name.split("_", 2)
        binom_name = leaf_name_split[0] + "_" + leaf_name_split[1]
        leaf.name = binom_name
    tip_set_diff = set(gene_tree2.get_leaf_names()) - set(species_tree.get_leaf_names())
    if tip_set_diff:
        sys.stderr.write(f"Warning. A total of {len(tip_set_diff)} species are missing in the species tree: {str(tip_set_diff)}\n")
    if is_ultrametric:
        cn = ["numerical_label", "spnode_coverage", 'spnode_age']
    else:
        cn = ["numerical_label", "spnode_coverage"]
    species_nodes = list(species_tree.traverse(strategy="postorder"))
    species_names = {sn: sn.name.replace('\'', '') for sn in species_nodes}
    species_leaf_node = {leaf.name: leaf for leaf in species_tree.iter_leaves()}
    species_depth = {}
    for sn in species_tree.traverse(strategy="preorder"):
        species_depth[sn] = 0 if sn.is_root() else (species_depth[sn.up] + 1)
    if is_ultrametric:
        species_age = {}
        for sn in species_nodes:
            if sn.is_leaf():
                species_age[sn] = 0.0
            else:
                first_child = sn.children[0]
                species_age[sn] = species_age[first_child] + first_child.dist
        species_up_age = {}
        for sn in species_nodes:
            if sn.is_root():
                species_up_age[sn] = numpy.inf
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
        if gn.is_leaf():
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
            "numerical_label": gn.numerical_label,
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
    return pandas.DataFrame(rows, columns=cn)

def ou2table(regime_file, leaf_file, input_tree_file):
    df_regime = pandas.read_csv(regime_file, sep="\t")
    df_leaf = pandas.read_csv(leaf_file, sep="\t")
    tree = ete3.PhyloNode(input_tree_file, format=1)
    tree = add_numerical_node_labels(tree)
    nodes = list(tree.traverse())
    num_nodes = len(nodes)
    tissues = df_leaf.columns[3:].values
    if 'expectations' in df_leaf['param'].values:
        df_leaf.loc[(df_leaf['param'] == 'expectations'), 'param'] = 'mu'
    cn1 = ["numerical_label", "regime", "is_shift", "num_child_shift"]
    cn2 = ["tau", "delta_tau", "delta_maxmu", "mu_complementarity"]
    cn3 = ["mu_" + tissue for tissue in tissues]
    cn = cn1 + cn2 + cn3
    regime_map = {}
    regime_rows = df_regime.loc[:, ["node_name", "regime"]].copy()
    regime_rows["node_name"] = regime_rows["node_name"].fillna(value="placeholder_text")
    for node_name, regime in regime_rows.itertuples(index=False, name=None):
        if pandas.isna(regime):
            continue
        if node_name not in regime_map:
            regime_map[node_name] = int(regime)
    for node in tree.traverse(strategy="preorder"):
        if node.is_root():
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

    numerical_label = numpy.empty(num_nodes, dtype=numpy.int64)
    regime = numpy.empty(num_nodes, dtype=numpy.int64)
    is_shift = numpy.empty(num_nodes, dtype=numpy.int64)
    num_child_shift = numpy.full(num_nodes, numpy.nan, dtype=float)
    mu_values = numpy.empty((num_nodes, len(cn3)), dtype=float)
    parent_labels = numpy.empty(num_nodes, dtype=numpy.int64)
    shift_pairs = []
    for row_idx, node in enumerate(nodes):
        node_label = node.numerical_label
        shift_flag = int((not node.is_root()) and (node.regime != node.up.regime))
        numerical_label[row_idx] = node_label
        regime[row_idx] = node.regime
        is_shift[row_idx] = shift_flag
        mu_values[row_idx, :] = mu_by_regime[node.regime]
        parent_labels[row_idx] = -1 if node.is_root() else node.up.numerical_label
        if not node.is_leaf():
            children = node.get_children()
            num_child_shift[row_idx] = int(node.regime != children[0].regime) + int(node.regime != children[1].regime)
        if shift_flag:
            shift_pairs.append((node_label, node.get_sisters()[0].numerical_label))

    df = pandas.DataFrame(
        {
            "numerical_label": numerical_label,
            "regime": regime,
            "is_shift": is_shift,
            "num_child_shift": num_child_shift,
        }
    )
    for col_idx, col in enumerate(cn3):
        df[col] = mu_values[:, col_idx]
    tau_values = calc_tau(df, cn3, unlog2=True, unPlus1=True)
    df["tau"] = tau_values
    tau_by_label = numpy.empty(num_nodes, dtype=float)
    tau_by_label[numerical_label] = tau_values
    parent_labels_safe = parent_labels.copy()
    parent_labels_safe[parent_labels_safe == -1] = 0
    delta_tau = tau_values - tau_by_label[parent_labels_safe]
    delta_tau[parent_labels == -1] = numpy.nan
    df["delta_tau"] = delta_tau

    mu_max_values = mu_values.max(axis=1)
    mu_unlog_values = numpy.clip(numpy.exp2(mu_values) - 1, a_min=0, a_max=None)
    label_to_idx = numpy.empty(num_nodes, dtype=numpy.int64)
    label_to_idx[numerical_label] = numpy.arange(num_nodes, dtype=numpy.int64)
    delta_maxmu = numpy.full(num_nodes, numpy.nan, dtype=float)
    mu_complementarity = numpy.full(num_nodes, numpy.nan, dtype=float)
    for my_label, sis_label in shift_pairs:
        my_idx = label_to_idx[my_label]
        sis_idx = label_to_idx[sis_label]
        delta_maxmu[my_idx] = float(mu_max_values[my_idx] - mu_max_values[sis_idx])
        mu_complementarity[my_idx] = calc_complementarity(mu_unlog_values[my_idx], mu_unlog_values[sis_idx])
    df["delta_maxmu"] = delta_maxmu
    df["mu_complementarity"] = mu_complementarity
    return df.loc[:, cn]

def get_misc_node_statistics(tree_file, tax_annot=False):
    tree = ete3.PhyloNode(tree_file, format=1)
    tree = add_numerical_node_labels(tree)
    cn1 = ["numerical_label", "taxon", "taxid", "num_sp", "num_leaf", "so_event", "dup_conf_score"]
    cn2 = ["parent", "sister", "child1", "child2", "so_event_parent"]
    cn = cn1 + cn2
    nodes = list(tree.traverse())
    if tax_annot:
        tree = taxonomic_annotation(tree)
    else:
        for node in nodes:
            node.taxid = -999
            if node.is_leaf():
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
        if node.is_leaf():
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
            child1_mask = species_mask_by_node[children[0]]
            child2_mask = species_mask_by_node[children[1]]
            union_mask = child1_mask | child2_mask
            union_count = union_mask.bit_count()
            if union_count == 0:
                dup_conf_score_by_node[node] = 0.0
            else:
                dup_conf_score_by_node[node] = (child1_mask & child2_mask).bit_count() / union_count
        else:
            dup_conf_score_by_node[node] = 0.0

    n_nodes = len(nodes)
    numerical_label = numpy.empty(n_nodes, dtype=numpy.int64)
    taxon = numpy.empty(n_nodes, dtype=object)
    taxid = numpy.empty(n_nodes, dtype=numpy.int64)
    num_sp = numpy.empty(n_nodes, dtype=numpy.int64)
    num_leaf = numpy.empty(n_nodes, dtype=numpy.int64)
    so_event = numpy.full(n_nodes, "L", dtype=object)
    dup_conf_score = numpy.zeros(n_nodes, dtype=float)
    parent = numpy.full(n_nodes, -999, dtype=numpy.int64)
    sister = numpy.full(n_nodes, -999, dtype=numpy.int64)
    child1 = numpy.full(n_nodes, -999, dtype=numpy.int64)
    child2 = numpy.full(n_nodes, -999, dtype=numpy.int64)
    so_event_parent = numpy.full(n_nodes, "S", dtype=object)

    for row_idx, node in enumerate(nodes):
        label = node.numerical_label
        node_dup_conf_score = dup_conf_score_by_node.get(node, 0.0)
        numerical_label[row_idx] = label
        taxon[row_idx] = str(node.sci_name)
        taxid[row_idx] = node.taxid
        num_sp[row_idx] = species_mask_by_node[node].bit_count()
        num_leaf[row_idx] = num_leaf_by_node[node]
        if hasattr(node.up, "numerical_label"):
            parent[row_idx] = node.up.numerical_label
        if node.up is not None:
            siblings = node.up.children
            if len(siblings) == 2:
                sister_node = siblings[1] if siblings[0] is node else siblings[0]
                sister[row_idx] = sister_node.numerical_label
            else:
                sister_nodes = node.get_sisters()
                if len(sister_nodes) == 1:
                    sister[row_idx] = sister_nodes[0].numerical_label
        if not node.is_leaf():
            if len(node.children) >= 1:
                child1[row_idx] = node.children[0].numerical_label
            if len(node.children) >= 2:
                child2[row_idx] = node.children[1].numerical_label
            dup_conf_score[row_idx] = node_dup_conf_score
            so_event[row_idx] = "D" if node_dup_conf_score > 0 else "S"
        if (node.up is not None) and (dup_conf_score_by_node.get(node.up, 0.0) > 0):
            so_event_parent[row_idx] = "D"

    return pandas.DataFrame(
        {
            "numerical_label": numerical_label,
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
    out = df.copy()
    parent_column = f'parent_{column}'
    value_by_label = out.set_index('numerical_label')[column]
    out[parent_column] = out['parent'].map(value_by_label)
    out[f'delta_{column}'] = out[column] - out[parent_column]
    out = out.drop(parent_column, axis=1)
    return out

def get_notung_root_stats(file):
    out = {}
    with open(file) as f:
        for line in f:
            if 'Number of optimal roots' in line:
                m = NOTUNG_OPT_ROOT_RE.search(line)
                if m is not None:
                    out['ntg_num_opt_root'] = int(m.group(1))
            if 'Best rooting score:' in line:
                m = NOTUNG_BEST_SCORE_RE.search(line)
                if m is not None:
                    out['ntg_best_root_score'] = float(m.group(1))
                    out['ntg_worst_root_score'] = float(m.group(2))
    return out

def get_notung_reconcil_stats(file):
    out = {}
    with open(file) as f:
        it = iter(f)
        for line in it:
            if 'Reconciliation Information' in line:
                dup_line = next(it, None)
                codiv_line = next(it, None)
                transfer_line = next(it, None)
                loss_line = next(it, None)
                if (dup_line is not None) and (codiv_line is not None) and (transfer_line is not None) and (loss_line is not None):
                    out['ntg_num_dup'] = int(dup_line.replace('- Duplications: ', ''))
                    out['ntg_num_codiv'] = int(codiv_line.replace('- Co-Divergences: ', ''))
                    out['ntg_num_transfer'] = int(transfer_line.replace('- Transfers: ', ''))
                    out['ntg_num_loss'] = int(loss_line.replace('- Losses: ', ''))
            if 'Tree Without Losses' in line:
                _ = next(it, None)
                _ = next(it, None)
                _ = next(it, None)
                polytomy_line = next(it, None)
                if polytomy_line is not None:
                    out['ntg_num_polytomy'] = int(polytomy_line.replace('- Polytomies: ', ''))
    return out


def get_root_stats(file):
    out = {}
    with open(file) as f:
        for line in f:
            if 'root positions with rho peak' in line:
                out['num_rho_peak'] = line.replace(ROOT_POSITIONS_PREFIX, '').count(' ')
            if 'Returning the' in line:
                m = ROOT_RETURNING_RE.search(line)
                if m is not None:
                    out['rooting_method'] = m.group(1).replace('first ', '')
    return out

def get_aln_stats(file):
    out = {}
    seq_lens_w_gap = []
    seq_lens = []
    seq_w_gap_len = 0
    seq_len = 0
    has_sequence = False
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
            seq_w_gap_len += len(seq_line)
            seq_len += len(seq_line) - seq_line.count('-')
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
    with gzip.open(file, 'rb') as f:
        for line in f:
            decoded = line.decode()
            if 'best_model_AIC:' in decoded:
                out['iqtree_best_AIC'] = decoded.replace('best_model_AIC: ', '').replace('\n', '')
            if 'best_model_AICc:' in decoded:
                out['iqtree_best_AICc'] = decoded.replace('best_model_AICc: ', '').replace('\n', '')
            if 'best_model_BIC:' in decoded:
                out['iqtree_best_BIC'] = decoded.replace('best_model_BIC: ', '').replace('\n', '')
    return out



def regime2tree(file):
    df = pandas.read_csv(file, sep='\t', header=0, index_col=False)
    out = {}
    out['num_regime'] = int(df['regime'].fillna(0).max() + 1)
    param_rows = df.loc[df['regime'].isnull(), :]
    params_set = set(param_rows.loc[:, 'param'].values)
    traits = list(df.columns[3:])
    dedup = param_rows.drop_duplicates(subset='param', keep='first')
    rows = dedup.loc[:, ['param'] + traits].to_numpy()
    for row in rows:
        param = row[0]
        out.update({f'{param}_{trait}': value for trait, value in zip(traits, row[1:])})
    if all(key in params_set for key in ['alpha', 'sigma2']):
        for trait in traits:
            out['gamma_' + trait] = out['sigma2_' + trait] / (2 * out['alpha_' + trait])
    return out


def get_dating_method(file):
    with open(file) as f:
        return f.read().replace('\n', '')


def get_most_recent(b, nl, og, target_col, target_value, return_col, og_col='orthogroup'):
    b_og = b.loc[b[og_col] == og, ['numerical_label', 'parent', target_col, return_col]]
    b_og = b_og.drop_duplicates(subset='numerical_label', keep='first').set_index('numerical_label', drop=False)
    root_nl = b_og.index.max()
    current_nl = nl
    while current_nl != root_nl:
        current_value = b_og.at[current_nl, target_col]
        if current_value == target_value:
            return b_og.at[current_nl, return_col]
        else:
            current_nl = b_og.at[current_nl, 'parent']
    return numpy.nan  # No target event found between the nl node and the root
