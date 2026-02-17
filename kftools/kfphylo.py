import sys

import ete3
import numpy


def get_tree_height(tree_file):
    tree = ete3.PhyloNode(tree_file, format=1)
    return tree.get_distance(target=tree.get_leaves()[0])


def transfer_internal_node_names(tree_to, tree_from):
    rf_dist = tree_to.robinson_foulds(tree_from)[0]
    assert rf_dist == 0, 'tree topologies are different. RF distance =' + str(rf_dist)
    tree_to = add_numerical_node_labels(tree_to)
    tree_from = add_numerical_node_labels(tree_from)
    name_by_label = {}
    for node in tree_from.traverse():
        if not node.is_leaf():
            name_by_label[node.numerical_label] = node.name
    for node in tree_to.traverse():
        if node.is_leaf():
            continue
        matched_name = name_by_label.get(node.numerical_label)
        if matched_name is not None:
            node.name = matched_name
    return tree_to


def fill_internal_node_names(tree):
    counter = 1
    for node in tree.traverse():
        if (not node.is_leaf()) and (node.name == ''):
            node.name = 'n' + str(counter)
            counter += 1
    return tree


def add_numerical_node_labels(tree):
    all_leaf_names = sorted(tree.get_leaf_names())
    leaf_numerical_labels = {leaf_name: (1 << i) for i, leaf_name in enumerate(all_leaf_names)}
    nodes = list(tree.traverse())
    node_label_sum = {}
    for node in tree.traverse(strategy="postorder"):
        if node.is_leaf():
            node_label_sum[node] = leaf_numerical_labels[node.name]
        else:
            node_mask = 0
            for child in node.get_children():
                node_mask |= node_label_sum[child]
            node_label_sum[node] = node_mask
    numerical_labels = [node_label_sum[node] for node in nodes]
    argsort_labels = numpy.argsort(numerical_labels)
    label_ranks = numpy.empty_like(argsort_labels)
    label_ranks[argsort_labels] = numpy.arange(len(argsort_labels))
    for i, node in enumerate(nodes):
        node.numerical_label = int(label_ranks[i])
    return tree


def transfer_root(tree_to, tree_from, verbose=False):
    tip_set_diff = set(tree_to.get_leaf_names()) - set(tree_from.get_leaf_names())
    if tip_set_diff:
        raise Exception('tree_to has more tips than tree_from. tip_set_diff = ' + str(tip_set_diff))
    subroot_leaves = [node.get_leaf_names() for node in tree_from.get_children()]
    is_n0_bigger_than_n1 = len(subroot_leaves[0]) > len(subroot_leaves[1])
    ingroups = subroot_leaves[0] if is_n0_bigger_than_n1 else subroot_leaves[1]
    outgroups = subroot_leaves[0] if not is_n0_bigger_than_n1 else subroot_leaves[1]
    if verbose:
        print('outgroups:', outgroups)
    tree_to.set_outgroup(ingroups[0])
    if len(outgroups) == 1:
        outgroup_ancestor = next(node for node in tree_to.iter_leaves() if node.name == outgroups[0])
    else:
        outgroup_ancestor = tree_to.get_common_ancestor(outgroups)
    tree_to.set_outgroup(outgroup_ancestor)
    subroot_to = tree_to.get_children()
    subroot_from = tree_from.get_children()
    total_subroot_length_to = sum(node.dist for node in subroot_to)
    total_subroot_length_from = sum(node.dist for node in subroot_from)
    dist_by_leafset = {
        frozenset(node.get_leaf_names()): node.dist
        for node in subroot_from
    }
    for n_to in subroot_to:
        n_to_leafset = frozenset(n_to.get_leaf_names())
        n_from_dist = dist_by_leafset.get(n_to_leafset)
        if n_from_dist is not None:
            n_to.dist = (n_from_dist / total_subroot_length_from) * total_subroot_length_to

    for n_to in tree_to.traverse():
        if n_to.name == '':
            n_to.name = tree_to.name
            tree_to.name = 'Root'
            break
    return tree_to


def check_ultrametric(tree, tol=0):
    min_dist = numpy.inf
    max_dist = -numpy.inf
    min_dist_leaf = None
    max_dist_leaf = None
    stack = [(tree, 0.0)]
    while stack:
        node, distance_from_root = stack.pop()
        if node.is_leaf():
            if distance_from_root < min_dist:
                min_dist = distance_from_root
                min_dist_leaf = node.name
            if distance_from_root > max_dist:
                max_dist = distance_from_root
                max_dist_leaf = node.name
        else:
            for child in node.get_children():
                stack.append((child, distance_from_root + child.dist))

    if numpy.isinf(min_dist):
        min_dist = 0.0
        max_dist = 0.0
        min_dist_leaf = tree.name
        max_dist_leaf = tree.name

    if tol == 0:
        tol = max_dist * 0.001
    dif_tree_length = max_dist - min_dist
    is_ultrametric = dif_tree_length < tol
    if dif_tree_length > tol:
        sys.stderr.write('(max - min) root-to-tip path ({}) was bigger than tol ({}).\n'.format(dif_tree_length, tol))
        sys.stderr.write('min_dist_leaf = {:,} in {}\n'.format(min_dist, min_dist_leaf))
        sys.stderr.write('max_dist_leaf = {:,} in {}\n'.format(max_dist, max_dist_leaf))
    return is_ultrametric


def taxonomic_annotation(tree):
    ncbi = ete3.NCBITaxa()
    leaves = list(tree.iter_leaves())
    sci_names = []
    for leaf in leaves:
        leaf_name_split = leaf.name.split("_")
        binom_name = leaf_name_split[0] + " " + leaf_name_split[1]
        leaf.sci_name = binom_name
        sci_names.append(leaf.sci_name)
    name2id = ncbi.get_name_translator(names=list(set(sci_names)))
    for leaf in leaves:
        if len(name2id[leaf.sci_name]) > 1:
            print(leaf.sci_name, "has", len(name2id[leaf.sci_name]), "taxids.")
        leaf.taxid = name2id[leaf.sci_name][0]
    tree.annotate_ncbi_taxa(taxid_attr="taxid")
    return tree
