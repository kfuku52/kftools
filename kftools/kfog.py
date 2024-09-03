import copy
import re

import pandas

from kfexpression import *
from kfphylo import *
from kfutil import *


def nwk2table(tree, attr='', age=False, parent=False, sister=False):
    if (attr == 'support'):
        tree_format = 0
    else:
        tree_format = 1
    cn = ["numerical_label", attr]
    if type(tree) == str:
        tree = ete3.PhyloNode(tree, format=tree_format)
    elif type(tree) == ete3.PhyloNode:
        tree = tree
    tree = add_numerical_node_labels(tree)
    df = pandas.DataFrame(numpy.nan, index=range(0, len(list(tree.traverse()))), columns=cn)
    df['numerical_label'] = -999
    dtype = type(getattr(tree.get_children()[0], attr))
    df[attr] = df[attr].astype(dtype)
    row = 0
    for node in tree.traverse():
        df.loc[row, "numerical_label"] = node.numerical_label
        if attr in dir(node):
            df.at[row, attr] = getattr(node, attr)
        row += 1
    if (attr == 'dist') & (age):
        assert check_ultrametric(tree)
        df['age'] = numpy.nan
        for node in tree.traverse():
            is_nn = (df['numerical_label'] == node.numerical_label)
            df.loc[is_nn, 'age'] = node.get_distance(target=node.get_leaves()[0])
    if parent:
        df.loc[:,'parent'] = -1
        for node in tree.traverse():
            if not node.is_root():
                is_nn = (df['numerical_label'] == node.numerical_label)
                df.loc[is_nn, 'parent'] = node.up.numerical_label
    if sister:
        df.loc[:,'sister'] = -1
        for node in tree.traverse():
            if not node.is_root():
                is_nn = (df['numerical_label'] == node.numerical_label)
                df.loc[is_nn, 'sister'] = node.get_sisters()[0].numerical_label
    df = df.sort_values(by='numerical_label', ascending=True).reset_index(drop=True)
    return df

def node_gene2species(gene_tree, species_tree, is_ultrametric=False):
    gene_tree2 = copy.deepcopy(gene_tree)
    gene_tree2 = add_numerical_node_labels(gene_tree2)
    if is_ultrametric:
        assert check_ultrametric(gene_tree2)
    for leaf in gene_tree2.iter_leaves():
        leaf_name_split = leaf.name.split("_")
        binom_name = leaf_name_split[0] + "_" + leaf_name_split[1]
        leaf.name = binom_name
    tip_set_diff = set(gene_tree2.get_leaf_names()) - set(species_tree.get_leaf_names())
    if len(tip_set_diff) != 0:
        sys.stderr.write(f"Warning. A total of {len(tip_set_diff)} species are missing in the species tree: {str(tip_set_diff)}\n")
    if is_ultrametric:
        cn = ["numerical_label", "spnode_coverage", 'spnode_age']
    else:
        cn = ["numerical_label", "spnode_coverage"]
    df = pandas.DataFrame(0, index=range(0, len(list(gene_tree2.traverse()))), columns=cn)
    if is_ultrametric:
        df['spnode_age'] = ''
        df['spnode_coverage'] = ''
    else:
        df['spnode_coverage'] = ''
    row = 0
    for gn in gene_tree2.traverse(strategy="postorder"):
        flag1 = 0
        flag2 = 0
        gnspp = set(gn.get_leaf_names())
        gn_age = gn.get_distance(target=gn.get_leaves()[0])
        df.at[row, "numerical_label"] = gn.numerical_label
        for sn in species_tree.traverse(strategy="postorder"):
            snspp = set(sn.get_leaf_names())
            sn_age = sn.get_distance(target=sn.get_leaves()[0])
            if sn.is_root():
                sn_up_age = numpy.inf
            else:
                sn_up_age = sn.up.get_distance(target=sn.up.get_leaves()[0])
            if (len(gnspp - snspp) == 0) & (flag1 == 0):
                df.at[row, 'spnode_coverage'] = sn.name.replace('\'', '')
                flag1 = 1
            if (gn_age >= sn_age) & (gn_age < sn_up_age) & (len(gnspp - snspp) == 0) & (flag2 == 0) & (is_ultrametric):
                df.at[row, 'spnode_age'] = sn.name.replace('\'', '')
                flag2 = 1
            if is_ultrametric:
                if (flag1 == 1) & (flag2 == 1):
                    break
            else:
                if (flag1 == 1):
                    break
        row += 1
    return df

def ou2table(regime_file, leaf_file, input_tree_file):
    df_regime = pandas.read_csv(regime_file, sep="\t")
    df_leaf = pandas.read_csv(leaf_file, sep="\t")
    tree = ete3.PhyloNode(input_tree_file, format=1)
    tree = add_numerical_node_labels(tree)
    tissues = df_leaf.columns[3:].values
    if ('expectations' in df_leaf['param'].values):
        df_leaf.loc[(df_leaf['param'] == 'expectations'), 'param'] = 'mu'
    cn1 = ["numerical_label", "regime", "is_shift", "num_child_shift"]
    cn2 = ["tau", "delta_tau", "delta_maxmu", "mu_complementarity"]
    cn3 = ["mu_" + tissue for tissue in tissues]
    cn = cn1 + cn2 + cn3
    df = pandas.DataFrame(0, index=range(0, len(list(tree.traverse()))), columns=cn)
    for node in tree.traverse():
        node.regime = 0 # initialize
    for node in tree.traverse():
        if node.name in df_regime.loc[:, "node_name"].fillna(value="placeholder_text").values:
            regime_nos = df_regime.loc[df_regime.node_name.values == node.name, "regime"]
            regime_no = int(regime_nos.iloc[0])
            for sub_node in node.traverse():
                sub_node.regime = regime_no
    is_mu = (df_leaf['param'] == 'mu')
    is_cols = [c not in ['node_name', 'param'] for c in df_leaf.columns]
    df_leaf_unique = df_leaf.loc[is_mu, df_leaf.columns[is_cols]]
    df_leaf_unique = df_leaf_unique.drop_duplicates()
    df_leaf_unique = df_leaf_unique.groupby(by='regime').mean()
    df_leaf_unique = df_leaf_unique.reset_index()
    for node in tree.traverse():
        node.mu = df_leaf_unique.loc[(df_leaf_unique['regime'] == node.regime), :]
    row = 0
    for node in tree.traverse():
        df.loc[row, "numerical_label"] = node.numerical_label
        df.loc[row, "regime"] = node.regime
        df.loc[row, cn3] = node.mu.loc[:, tissues].values[0]
        is_shift = 0
        if not node.is_root():
            if node.regime != node.up.regime:
                is_shift = 1
        df.loc[row, "is_shift"] = is_shift
        row += 1
    df["tau"] = calc_tau(df, cn3, unlog2=True, unPlus1=True)
    row = 0
    for node in tree.traverse():
        # highest_value = df.loc[df.numerical_label==node.numerical_label,cn3].max(axis=1).values
        # is_highest = df.loc[df.numerical_label==node.numerical_label,cn3].values.reshape(-1) == numpy.float(highest_value)
        # highest_in = numpy.array(cn3)[is_highest][0].replace("mu_", "")
        # df.loc[df.numerical_label==node.numerical_label,"highest_mu"] = highest_in
        if not node.is_root():
            tau_up = df.loc[df.numerical_label == node.up.numerical_label, "tau"].values
            tau_my = df.loc[df.numerical_label == node.numerical_label, "tau"].values
            df.loc[df.numerical_label == node.numerical_label, "delta_tau"] = tau_my - tau_up
            if df.loc[df.numerical_label == node.numerical_label, "is_shift"].values:
                my_label = node.numerical_label
                sis_label = node.get_sisters()[0].numerical_label
                my_maxmu = df.loc[df.numerical_label == my_label, cn3].max(axis=1).values
                sis_maxmu = df.loc[df.numerical_label == sis_label, cn3].max(axis=1).values
                delta_maxmu = my_maxmu - sis_maxmu
                df.loc[df.numerical_label == node.numerical_label, "delta_maxmu"] = delta_maxmu
                my_mu = df.loc[df.numerical_label == my_label, cn3]
                sis_mu = df.loc[df.numerical_label == sis_label, cn3]
                my_mu_unlog = (numpy.exp2(my_mu) - 1).clip(lower=0).values[0]
                sis_mu_unlog = (numpy.exp2(sis_mu) - 1).clip(lower=0).values[0]
                df.loc[df.numerical_label == node.numerical_label, "mu_complementarity"] = calc_complementarity(
                    my_mu_unlog, sis_mu_unlog)
        if not node.is_leaf():
            is_child1_shift = (node.regime != node.get_children()[0].regime)
            is_child2_shift = (node.regime != node.get_children()[1].regime)
            num_child_shift = sum([is_child1_shift, is_child2_shift])
            df.loc[row, "num_child_shift"] = num_child_shift
        row += 1
    return (df)

def get_misc_node_statistics(tree_file, tax_annot=False):
    tree = ete3.PhyloNode(tree_file, format=1)
    tree = add_numerical_node_labels(tree)
    cn1 = ["numerical_label", "taxon", "taxid", "num_sp", "num_leaf", "so_event", "dup_conf_score"]
    cn2 = ["parent", "sister", "child1", "child2", "so_event_parent"]
    cn = cn1 + cn2
    df = pandas.DataFrame(0, index=range(0, len(list(tree.traverse()))), columns=cn)
    df.loc[:, 'dup_conf_score'] = df.loc[:, 'dup_conf_score'].astype(float)
    df.loc[:, "parent"] = -999
    df.loc[:, "sister"] = -999
    df.loc[:, "child1"] = -999
    df.loc[:, "child2"] = -999
    df.loc[:, "so_event"] = "L"
    df.loc[:, "so_event_parent"] = "S"
    df['taxon'] = df['taxon'].astype('str')
    if tax_annot:
        tree = taxonomic_annotation(tree)
    else:
        for node in tree.traverse():
            node.taxid = -999
            if node.is_leaf():
                node.sci_name = re.sub('_.*','',node.name.replace('_',' ',1))
            else:
                node.sci_name = ''
    row = 0
    for node in tree.traverse():
        df.at[row, "numerical_label"] = node.numerical_label
        df.at[row, "taxon"] = node.sci_name
        df.at[row, "taxid"] = node.taxid
        df.at[row, "num_sp"] = len(set([leaf.sci_name for leaf in node.iter_leaves()]))
        df.at[row, "num_leaf"] = len(list(node.get_leaves()))
        if hasattr(node.up, "numerical_label"):
            df.at[row, "parent"] = node.up.numerical_label
        sister = node.get_sisters()
        if len(sister) == 1:
            df.at[row, "sister"] = sister[0].numerical_label
        if not node.is_leaf():
            df.at[row, "child1"] = node.children[0].numerical_label
            df.at[row, "child2"] = node.children[1].numerical_label
            sp_child1 = set([leaf.sci_name for leaf in node.children[0].iter_leaves()])
            sp_child2 = set([leaf.sci_name for leaf in node.children[1].iter_leaves()])
            num_union = len(sp_child1.union(sp_child2))
            num_intersection = len(sp_child1.intersection(sp_child2))
            node.dup_conf_score = num_intersection / num_union
            df.at[row, "dup_conf_score"] = node.dup_conf_score
            if node.dup_conf_score > 0:
                df.at[row, "so_event"] = "D"
            elif node.dup_conf_score == 0:
                df.at[row, "so_event"] = "S"
        if not isinstance(node.up, type(None)):
            if (node.up.dup_conf_score > 0):
                df.at[row, "so_event_parent"] = "D"
        row += 1
    return (df)

def compute_delta(df, column):
    df_parent = df.loc[:, ['numerical_label', column]]
    df_parent.columns = ['parent', 'parent_' + column]
    df = pandas.merge(df, df_parent, on='parent', how='left')
    df['delta_' + column] = df[column] - df['parent_' + column]
    df = df.drop('parent_' + column, axis=1)
    return df

def get_notung_root_stats(file):
    out = dict()
    with open(file) as f:
        for l in f.readlines():
            if 'Number of optimal roots' in l:
                r = re.compile("Number of optimal roots: ([0-9]+) out of ([0-9]+)")
                m = r.search(l)
                out['ntg_num_opt_root'] = int(m.group(1))
            if 'Best rooting score:' in l:
                r = re.compile("Best rooting score: (\d*[.,]?\d*), worst rooting score: (\d*[.,]?\d*)")
                m = r.search(l)
                out['ntg_best_root_score'] = float(m.group(1))
                out['ntg_worst_root_score'] = float(m.group(2))

    return out

def get_notung_reconcil_stats(file):
    out = dict()
    with open(file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if ('Reconciliation Information' in lines[i]):
                out['ntg_num_dup'] = int(lines[i + 1].replace('- Duplications: ', ''))
                out['ntg_num_codiv'] = int(lines[i + 2].replace('- Co-Divergences: ', ''))
                out['ntg_num_transfer'] = int(lines[i + 3].replace('- Transfers: ', ''))
                out['ntg_num_loss'] = int(lines[i + 4].replace('- Losses: ', ''))
            if ('Tree Without Losses' in lines[i]):
                out['ntg_num_polytomy'] = int(lines[i + 4].replace('- Polytomies: ', ''))
    return out


def get_root_stats(file):
    out = dict()
    with open(file) as f:
        for l in f.readlines():
            if 'root positions with rho peak' in l:
                out['num_rho_peak'] = l.replace('root positions with rho peak: ', '').count(' ')
            if 'Returning the' in l:
                r = re.compile("Returning the (.*) tree")
                m = r.search(l)
                rooting_method = m.group(1).replace('first ', '')
                out['rooting_method'] = rooting_method
    return out

def get_aln_stats(file):
    out = dict()
    seqs = open(file).read().split('>')
    for i in range(len(seqs)):
        seqs[i] = re.sub('^[^\n]*\n', '', seqs[i])
    seq_lens_w_gap = [len(seq) for seq in seqs if len(seq) != 0]
    out['num_site'] = max(seq_lens_w_gap)
    for i in range(len(seqs)):
        seqs[i] = seqs[i].replace('\n', '').replace('-', '')
    seq_lens = [len(seq) for seq in seqs if len(seq) != 0]
    out['num_seq'] = len(seq_lens)
    out['len_max'] = max(seq_lens)
    out['len_min'] = min(seq_lens)
    return out


def get_iqtree_model_stats(file):
    out = dict()
    import gzip
    with gzip.open(file, 'rb') as f:
        for l in f.readlines():
            l = l.decode()
            if 'best_model_AIC:' in l:
                out['iqtree_best_AIC'] = l.replace('best_model_AIC: ', '').replace('\n', '')
            if 'best_model_AICc:' in l:
                out['iqtree_best_AICc'] = l.replace('best_model_AICc: ', '').replace('\n', '')
            if 'best_model_BIC:' in l:
                out['iqtree_best_BIC'] = l.replace('best_model_BIC: ', '').replace('\n', '')
    return out



def regime2tree(file):
    df = pandas.read_csv(file, sep='\t', header=0, index_col=False)
    out = dict()
    out['num_regime'] = int(df['regime'].fillna(0).max()+1)
    params = df.loc[df['regime'].isnull(), 'param'].values
    traits = df.columns[3:]
    for param in params:
        for trait in traits:
            out[param + '_' + trait] = df.loc[(df['param'] == param), trait].values[0]
    if (all([key in params for key in ['alpha', 'sigma2']])):
        for trait in traits:
            out['gamma_' + trait] = out['sigma2_' + trait] / (2 * out['alpha_' + trait])
    return out


def get_dating_method(file):
    with open(file) as f:
        dating_method = f.read().replace('\n', '')
    return dating_method


def get_most_recent(b, nl, og, target_col, target_value, return_col, og_col='orthogroup'):
    is_og = (b.loc[:,og_col]==og)
    b_og = b.loc[is_og,:]
    root_nl = b_og.loc[:,'numerical_label'].max()
    current_nl = nl
    while (current_nl!=root_nl):
        is_current_nl = (b_og.loc[:,'numerical_label']==current_nl)
        current_value = b_og.loc[is_current_nl,target_col].values[0]
        if (current_value==target_value):
            return b_og.loc[is_current_nl,return_col].values[0]
        else:
            current_nl = b_og.loc[is_current_nl,'parent'].values[0]
    return numpy.nan # No target event found between the nl node and the root
