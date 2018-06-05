import numpy, ete3, re

def get_tree_height(tree_file):
    tree = ete3.PhyloNode(tree_file, format=1)
    height = tree.get_distance(target=tree.get_leaves()[0])
    return height

def transfer_internal_node_names(tree_to, tree_from):
    rf_dist = tree_to.robinson_foulds(tree_from)[0]
    assert rf_dist==0, 'tree topologies are different. RF distance ='+str(rf_dist)
    for to in tree_to.traverse():
        if not to.is_leaf():
            for fr in tree_from.traverse():
                if not fr.is_leaf():
                    if set(to.get_leaf_names())==set(fr.get_leaf_names()):
                        to.name = fr.name
    return tree_to

def fill_internal_node_names(tree):
    counter = 1
    for n in tree.traverse():
        if not n.is_leaf():
            if n.name=='':
                n.name = 'n'+str(counter)
                counter += 1
    return tree

def add_numerical_node_labels(tree):
    all_leaf_names = tree.get_leaf_names()
    all_leaf_names.sort()
    leaf_numerical_labels = dict()
    power = 0
    for i in range(0, len(all_leaf_names)):
        leaf_numerical_labels[all_leaf_names[i]] = 2**i
    numerical_labels = list()
    for node in tree.traverse():
        leaf_names = node.get_leaf_names()
        numerical_labels.append(sum([leaf_numerical_labels[leaf_name] for leaf_name in leaf_names]))
    argsort_labels = numpy.argsort(numerical_labels)
    short_labels = numpy.arange(len(argsort_labels))
    i=0
    for node in tree.traverse():
        node.numerical_label = short_labels[argsort_labels==i][0]
        i+=1
    return(tree)

def transfer_root(tree_to, tree_from, verbose=False):
    assert len(set(tree_to.get_leaf_names()) - set(tree_from.get_leaf_names())) == 0
    subroot_leaves = [ n.get_leaf_names() for n in tree_from.get_children() ]
    is_n0_bigger_than_n1 = (len(subroot_leaves[0]) > len(subroot_leaves[1]))
    ingroups = subroot_leaves[0] if is_n0_bigger_than_n1 else subroot_leaves[1]
    outgroups = subroot_leaves[0] if not is_n0_bigger_than_n1 else subroot_leaves[1]
    if verbose:
        print('outgroups:', outgroups)
    tree_to.set_outgroup(ingroups[0])
    if (len(outgroups) == 1):
        outgroup_ancestor = [n for n in tree_to.iter_leaves() if n.name == outgroups[0]][0]
    else:
        outgroup_ancestor = tree_to.get_common_ancestor(outgroups)
    tree_to.set_outgroup(outgroup_ancestor)
    subroot_node_names = [n.name for n in tree_to.get_children()]
    subroot_to = tree_to.get_children()
    subroot_from = tree_from.get_children()
    total_subroot_length_to = sum([n.dist for n in subroot_to])
    total_subroot_length_from = sum([n.dist for n in subroot_from])
    for n_to in subroot_to:
        for n_from in subroot_from:
            if (set(n_to.get_leaf_names()) == set(n_from.get_leaf_names())):
                n_to.dist = (n_from.dist / total_subroot_length_from) * total_subroot_length_to
    for n_to in tree_to.traverse():
        if n_to.name == '':
            n_to.name = tree_to.name
            tree_to.name = 'Root'
            break
    return tree_to

def check_ultrametric(tree, tol=1e-3):
    root2leaf_dist = [tree.get_distance(target=l) for l in tree.get_leaves()]
    is_ultrametric = (max(root2leaf_dist) - min(root2leaf_dist)) < tol
    return is_ultrametric

def taxonomic_annotation(tree):
    ncbi = ete3.NCBITaxa()
    for leaf in tree.iter_leaves():
        leaf_name_split = leaf.name.split("_")
        binom_name = leaf_name_split[0] + " " + leaf_name_split[1]
        leaf.sci_name = binom_name
        name2id = ncbi.get_name_translator(names=[leaf.sci_name])
        if len(name2id[leaf.sci_name]) > 1:
            print(leaf.sci_name, "has", len(name2id[leaf.sci_name]), "taxids.")
        leaf.taxid = name2id[leaf.sci_name][0]
    tax2names, tax2lineages, tax2rank = tree.annotate_ncbi_taxa(taxid_attr="taxid")
    return (tree)
