import re

NUCLEOTIDES = ('A', 'T', 'C', 'G')
CODON_POSITIONS = (0, 1, 2)


def codon2nuc_freqs(codon_freqs=None, model=''):
    if codon_freqs is None:
        codon_freqs = {}
    if 'F1X4' in model:
        nuc_freqs = [{nuc: 0 for nuc in NUCLEOTIDES}]
        for codon, codon_freq in codon_freqs.items():
            for nuc in NUCLEOTIDES:
                nuc_count = sum(nuc == c for c in codon)
                nuc_freqs[0][nuc] += codon_freq * nuc_count / 3
    elif 'F3X4' in model:
        nuc_freqs = [{nuc: 0 for nuc in NUCLEOTIDES} for _ in CODON_POSITIONS]
        for codon_pos in CODON_POSITIONS:
            for codon, codon_freq in codon_freqs.items():
                nuc_freqs[codon_pos][codon[codon_pos]] += codon_freq / 3
    else:
        raise ValueError("model must contain either 'F1X4' or 'F3X4'")
    for freqs in nuc_freqs:
        scale_factor = sum(freqs.values())
        for nuc in freqs.keys():
            freqs[nuc] = freqs[nuc] / scale_factor
    return nuc_freqs


def nuc_freq2theta(nuc_freqs=None):
    if nuc_freqs is None:
        nuc_freqs = []
    thetas = []
    for freqs in nuc_freqs:
        AT_freq = freqs['A'] + freqs['T']
        GC_freq = freqs['G'] + freqs['C']
        theta = GC_freq
        if AT_freq != 0:
            theta1 = freqs['A'] / (freqs['A'] + freqs['T'])
        else:
            theta1 = 0.5
        if GC_freq != 0:
            theta2 = freqs['G'] / (freqs['G'] + freqs['C'])
        else:
            theta2 = 0.5
        thetas.append({'theta': theta, 'theta1': theta1, 'theta2': theta2})
    return thetas


def get_mapnh_thetas(model, thetas):
    model_frequency = model
    model_frequency = re.sub(r'X4\+.*', 'X4', model_frequency)
    model_frequency = re.sub(r'.*\+F', 'F', model_frequency)
    values = []
    for i, theta in enumerate(thetas):
        if len(thetas) == 1:
            values.append('Full.theta=' + str(theta['theta']))
            values.append('Full.theta1=' + str(theta['theta1']))
            values.append('Full.theta2=' + str(theta['theta2']))
        else:
            values.append(str(i + 1) + '_Full.theta=' + str(theta['theta']))
            values.append(str(i + 1) + '_Full.theta1=' + str(theta['theta1']))
            values.append(str(i + 1) + '_Full.theta2=' + str(theta['theta2']))
    if len(values) == 0:
        return model_frequency + '('
    return model_frequency + '(' + ','.join(values) + ')'


def alignment2nuc_freqs(leaf_name, alignment_file, model):
    seq = None
    seq_chunks = []
    in_target = False
    with open(alignment_file) as f:
        for line in f:
            if line.startswith('>'):
                if in_target and seq_chunks:
                    break
                header = line[1:].strip()
                in_target = header.startswith(leaf_name)
                if in_target:
                    seq_chunks = []
            elif in_target:
                seq_chunks.append(line.strip())
    if seq_chunks:
        seq = ''.join(seq_chunks)
    if seq is None:
        raise ValueError(f"leaf_name '{leaf_name}' was not found in alignment_file")
    if 'F1X4' in model:
        print('F1X4 is not yet implemented')
        raise NotImplementedError('F1X4 is not yet implemented')
    elif 'F3X4' in model:
        seq_codons = [seq[start::3] for start in CODON_POSITIONS]
        nuc_freqs = []
        for codon_seq in seq_codons:
            codon_nuc_freqs = {}
            for nuc in ['A', 'C', 'G', 'T']:
                codon_nuc_freqs[nuc] = codon_seq.count(nuc) / len(codon_seq)
            nuc_freqs.append(codon_nuc_freqs)
    else:
        raise ValueError("model must contain either 'F1X4' or 'F3X4'")
    return nuc_freqs


def weighted_mean_root_thetas(subroot_thetas, tree, model):
    subroot_nodes = tree.get_children()
    subroot_branch_lengths = {}
    for subroot_node in subroot_nodes:
        assert subroot_node.name in subroot_thetas.keys()
        subroot_branch_lengths[subroot_node.name] = subroot_node.dist
    if 'F1X4' in model:
        print('F1X4 is not yet implemented')
        raise NotImplementedError('F1X4 is not yet implemented')
    elif 'F3X4' in model:
        if len(subroot_nodes) == 2:
            left_node, right_node = subroot_nodes
            left_key = left_node.name
            right_key = right_node.name
            left_bl = left_node.dist
            right_bl = right_node.dist
            left_thetas = subroot_thetas[left_key]
            right_thetas = subroot_thetas[right_key]
            params = list(left_thetas[0].keys())
            root_thetas = []
            for codon_position in CODON_POSITIONS:
                codon_position_thetas = {}
                left_cp = left_thetas[codon_position]
                right_cp = right_thetas[codon_position]
                for param in params:
                    left_value = left_cp[param]
                    right_value = right_cp[param]
                    if left_value == right_value:
                        weighted_mean = left_value
                    elif left_value < right_value:
                        weighted_mean = left_value + ((right_value - left_value) * (left_bl / (left_bl + right_bl)))
                    else:
                        weighted_mean = right_value + ((left_value - right_value) * (right_bl / (right_bl + left_bl)))
                    codon_position_thetas[param] = weighted_mean
                root_thetas.append(codon_position_thetas)
            return root_thetas

        root_thetas = []
        params = list(list(subroot_thetas.values())[0][0].keys())
        subroot_items = [
            (subroot_theta, subroot_branch_lengths[subroot_key])
            for subroot_key, subroot_theta in subroot_thetas.items()
        ]
        for codon_position in CODON_POSITIONS:
            codon_position_thetas = {}
            for param in params:
                min_value = None
                max_value = None
                min_branch_length = None
                max_branch_length = None
                for subroot_theta, branch_length in subroot_items:
                    value = subroot_theta[codon_position][param]
                    if (min_value is None) or (value < min_value):
                        min_value = value
                        min_branch_length = branch_length
                    elif value == min_value:
                        min_branch_length = branch_length
                    if (max_value is None) or (value > max_value):
                        max_value = value
                        max_branch_length = branch_length
                    elif value == max_value:
                        max_branch_length = branch_length
                if min_value == max_value:
                    weighted_mean = min_value
                else:
                    weighted_mean = min_value + ((max_value - min_value) * (min_branch_length / (min_branch_length + max_branch_length)))
                codon_position_thetas[param] = weighted_mean
            root_thetas.append(codon_position_thetas)
    else:
        raise ValueError("model must contain either 'F1X4' or 'F3X4'")
    return root_thetas
