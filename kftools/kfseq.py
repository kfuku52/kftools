import re
import numbers
import os
from collections.abc import Mapping

import numpy as np
NUCLEOTIDES = ('A', 'T', 'C', 'G')
CODON_POSITIONS = (0, 1, 2)
CODON_LENGTH = 3


def _validate_model_string(model):
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    if model.strip() == "":
        raise ValueError("model must not be empty")


def _normalize_nucleotide_frequencies(freqs):
    scale_factor = sum(freqs.values())
    if (not np.isfinite(scale_factor)) or (scale_factor <= 0):
        raise ValueError('Nucleotide frequencies must have a positive total')
    for nuc in freqs.keys():
        freqs[nuc] = freqs[nuc] / scale_factor


def _canonicalize_codon_frequencies(codon_freqs):
    canonicalized = {}
    for codon, codon_freq in codon_freqs.items():
        if (not isinstance(codon, str)) or (len(codon) != CODON_LENGTH):
            raise ValueError('codon_freqs keys must be codon strings of length 3')
        if isinstance(codon_freq, bool) or (not isinstance(codon_freq, numbers.Real)) or (not np.isfinite(codon_freq)):
            raise ValueError('codon frequencies must be finite numeric values')
        if codon_freq < 0:
            raise ValueError('codon frequencies must be non-negative')
        codon_upper = codon.upper()
        invalid_nucleotides = sorted(set(codon_upper) - set(NUCLEOTIDES))
        if len(invalid_nucleotides) > 0:
            raise ValueError(
                f"codon '{codon}' contains invalid nucleotides: {invalid_nucleotides}"
            )
        canonicalized[codon_upper] = canonicalized.get(codon_upper, 0) + codon_freq
    return canonicalized


def _interpolate_by_branch_length(min_value, max_value, min_branch_length, max_branch_length):
    if min_value == max_value:
        return min_value
    if (min_branch_length < 0) or (max_branch_length < 0):
        raise ValueError('Branch lengths must be non-negative')
    denominator = min_branch_length + max_branch_length
    if denominator == 0:
        return (min_value + max_value) / 2
    return min_value + ((max_value - min_value) * (min_branch_length / denominator))


def _validate_nucleotide_frequency_dict(freqs):
    missing_nucleotides = sorted(set(NUCLEOTIDES) - set(freqs.keys()))
    if len(missing_nucleotides) > 0:
        raise ValueError(f"nucleotide frequency dictionary is missing keys: {missing_nucleotides}")
    for nuc in NUCLEOTIDES:
        value = freqs[nuc]
        if isinstance(value, bool) or (not isinstance(value, numbers.Real)) or (not np.isfinite(value)):
            raise ValueError(f"nucleotide frequency for '{nuc}' must be a finite numeric value")
        if value < 0:
            raise ValueError(f"nucleotide frequency for '{nuc}' must be non-negative")


def codon2nuc_freqs(codon_freqs=None, model=''):
    if codon_freqs is None:
        codon_freqs = {}
    if not isinstance(codon_freqs, Mapping):
        raise ValueError("codon_freqs must be a mapping from codon strings to frequencies")
    _validate_model_string(model)
    codon_freqs = _canonicalize_codon_frequencies(codon_freqs)
    if 'F1X4' in model:
        nuc_freqs = [{nuc: 0 for nuc in NUCLEOTIDES}]
        for codon, codon_freq in codon_freqs.items():
            for nuc in NUCLEOTIDES:
                nuc_count = sum(nuc == c for c in codon)
                nuc_freqs[0][nuc] += codon_freq * nuc_count / CODON_LENGTH
    elif 'F3X4' in model:
        nuc_freqs = [{nuc: 0 for nuc in NUCLEOTIDES} for _ in CODON_POSITIONS]
        for codon_pos in CODON_POSITIONS:
            for codon, codon_freq in codon_freqs.items():
                nuc_freqs[codon_pos][codon[codon_pos]] += codon_freq / CODON_LENGTH
    else:
        raise ValueError("model must contain either 'F1X4' or 'F3X4'")
    for freqs in nuc_freqs:
        _normalize_nucleotide_frequencies(freqs)
    return nuc_freqs


def nuc_freq2theta(nuc_freqs=None):
    if nuc_freqs is None:
        nuc_freqs = []
    if not isinstance(nuc_freqs, (list, tuple)):
        raise ValueError("nuc_freqs must be a list or tuple of nucleotide-frequency dictionaries")
    thetas = []
    for freqs in nuc_freqs:
        if not isinstance(freqs, dict):
            raise ValueError("each entry in nuc_freqs must be a dictionary")
        _validate_nucleotide_frequency_dict(freqs)
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
    _validate_model_string(model)
    if thetas is None:
        thetas = []
    if not isinstance(thetas, (list, tuple)):
        raise ValueError("thetas must be a list or tuple of theta dictionaries")
    required_theta_keys = {"theta", "theta1", "theta2"}
    for theta in thetas:
        if not isinstance(theta, dict):
            raise ValueError("each theta entry must be a dictionary")
        missing_keys = sorted(required_theta_keys - set(theta.keys()))
        if len(missing_keys) > 0:
            raise ValueError(f"theta entry is missing keys: {missing_keys}")
        for theta_key in required_theta_keys:
            theta_value = theta[theta_key]
            if isinstance(theta_value, bool) or (not isinstance(theta_value, numbers.Real)) or (not np.isfinite(theta_value)):
                raise ValueError(f"theta entry key '{theta_key}' must be a finite numeric value")
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
        return model_frequency + '()'
    return model_frequency + '(' + ','.join(values) + ')'


def alignment2nuc_freqs(leaf_name, alignment_file, model):
    _validate_model_string(model)
    if (not isinstance(leaf_name, str)) or (leaf_name.strip() == ""):
        raise ValueError("leaf_name must be a non-empty string")
    if isinstance(alignment_file, (bytes, bytearray)):
        raise ValueError("alignment_file must be a path-like string (bytes are not supported)")
    if not isinstance(alignment_file, (str, os.PathLike)):
        raise ValueError("alignment_file must be a path-like string")
    try:
        alignment_file = os.fspath(alignment_file)
    except TypeError as exc:
        raise ValueError("alignment_file must be a path-like string") from exc
    if isinstance(alignment_file, (bytes, bytearray)):
        raise ValueError("alignment_file must be a path-like string (bytes are not supported)")
    seq = None
    seq_chunks = []
    in_target = False
    found_target = False
    target_header_count = 0
    try:
        with open(alignment_file) as f:
            for line in f:
                if line.startswith('>'):
                    header = line[1:].strip()
                    header_id = header.split()[0] if header else ''
                    is_target_header = (header == leaf_name) or (header_id == leaf_name)
                    if is_target_header:
                        target_header_count += 1
                        if target_header_count > 1:
                            raise ValueError(
                                f"leaf_name '{leaf_name}' appears multiple times in alignment_file"
                            )
                        found_target = True
                        in_target = True
                        seq_chunks = []
                    else:
                        in_target = False
                elif in_target:
                    seq_chunks.append(line.strip())
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read alignment_file: {alignment_file}") from exc
    if seq_chunks:
        seq = ''.join(seq_chunks)
    if found_target and (seq is None):
        raise ValueError(f"Sequence for leaf '{leaf_name}' is empty in alignment_file")
    if seq is None:
        raise ValueError(f"leaf_name '{leaf_name}' was not found in alignment_file")
    seq = seq.upper()
    if 'F1X4' in model:
        raise NotImplementedError('F1X4 is not yet implemented')
    elif 'F3X4' in model:
        invalid_nucleotides = sorted(set(seq) - set(NUCLEOTIDES))
        if len(invalid_nucleotides) > 0:
            raise ValueError(
                f"Sequence for leaf '{leaf_name}' contains invalid nucleotides: {invalid_nucleotides}"
            )
        seq_codons = [seq[start::3] for start in CODON_POSITIONS]
        if any(len(codon_seq) == 0 for codon_seq in seq_codons):
            raise ValueError('F3X4 requires sequences with at least three nucleotides')
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
    _validate_model_string(model)
    if not isinstance(subroot_thetas, dict):
        raise ValueError("subroot_thetas must be a dictionary keyed by subroot node names")
    if tree is None:
        raise ValueError("tree must not be None")
    if not hasattr(tree, "get_children"):
        raise ValueError("tree must be an ete tree-like object with get_children()")
    subroot_nodes = tree.get_children()
    if len(subroot_nodes) == 0:
        raise ValueError("tree must contain at least one subroot child")
    subroot_names = [subroot_node.name for subroot_node in subroot_nodes]
    if any((name is None) or (name == "") for name in subroot_names):
        raise ValueError("All immediate subroot child nodes must have non-empty names")
    if len(set(subroot_names)) != len(subroot_names):
        raise ValueError("Immediate subroot child node names must be unique")
    expected_subroot_names = set(subroot_names)
    provided_subroot_names = set(subroot_thetas.keys())
    extra_subroot_names = sorted(provided_subroot_names - expected_subroot_names)
    if len(extra_subroot_names) > 0:
        raise ValueError(f"subroot_thetas contains unknown node names: {extra_subroot_names}")
    subroot_branch_lengths = {}
    for subroot_node in subroot_nodes:
        if subroot_node.name not in subroot_thetas:
            raise ValueError(f"subroot_thetas is missing node '{subroot_node.name}'")
        branch_length = subroot_node.dist
        if isinstance(branch_length, bool) or (not isinstance(branch_length, numbers.Real)) or (not np.isfinite(branch_length)):
            raise ValueError(
                f"Branch length for subroot node '{subroot_node.name}' must be a finite numeric value"
            )
        if branch_length < 0:
            raise ValueError(
                f"Branch length for subroot node '{subroot_node.name}' must be non-negative"
            )
        subroot_branch_lengths[subroot_node.name] = branch_length
    expected_num_positions = len(CODON_POSITIONS)
    for subroot_name in subroot_names:
        theta_by_position = subroot_thetas[subroot_name]
        if not isinstance(theta_by_position, (list, tuple)):
            raise ValueError(f"subroot_thetas['{subroot_name}'] must be a list/tuple of theta dictionaries")
        if len(theta_by_position) != expected_num_positions:
            raise ValueError(
                f"subroot_thetas['{subroot_name}'] must contain {expected_num_positions} codon-position entries"
            )
        for codon_position, theta_entry in enumerate(theta_by_position):
            if not isinstance(theta_entry, dict):
                raise ValueError(
                    f"subroot_thetas['{subroot_name}'][{codon_position}] must be a dictionary of theta parameters"
                )
            for param_name, param_value in theta_entry.items():
                if isinstance(param_value, bool) or (not isinstance(param_value, numbers.Real)) or (not np.isfinite(param_value)):
                    raise ValueError(
                        f"subroot_thetas['{subroot_name}'][{codon_position}]['{param_name}'] "
                        "must be a finite numeric value"
                    )
    reference_params = set(subroot_thetas[subroot_names[0]][0].keys())
    for subroot_name in subroot_names:
        for codon_position in CODON_POSITIONS:
            params_here = set(subroot_thetas[subroot_name][codon_position].keys())
            if params_here != reference_params:
                raise ValueError(
                    "All subroot theta dictionaries must share identical parameter keys across codon positions"
                )
    if 'F1X4' in model:
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
                        weighted_mean = _interpolate_by_branch_length(left_value, right_value, left_bl, right_bl)
                    else:
                        weighted_mean = _interpolate_by_branch_length(right_value, left_value, right_bl, left_bl)
                    codon_position_thetas[param] = weighted_mean
                root_thetas.append(codon_position_thetas)
            return root_thetas

        root_thetas = []
        params = list(list(subroot_thetas.values())[0][0].keys())
        subroot_items = [
            (subroot_thetas[subroot_name], subroot_branch_lengths[subroot_name])
            for subroot_name in subroot_names
        ]
        for codon_position in CODON_POSITIONS:
            codon_position_thetas = {}
            for param in params:
                value_and_branch_lengths = [
                    (subroot_theta[codon_position][param], branch_length)
                    for subroot_theta, branch_length in subroot_items
                ]
                values = [value for value, _ in value_and_branch_lengths]
                min_value = min(values)
                max_value = max(values)
                if min_value == max_value:
                    weighted_mean = min_value
                else:
                    min_branch_lengths = [
                        branch_length
                        for value, branch_length in value_and_branch_lengths
                        if value == min_value
                    ]
                    max_branch_lengths = [
                        branch_length
                        for value, branch_length in value_and_branch_lengths
                        if value == max_value
                    ]
                    min_branch_length = float(np.mean(min_branch_lengths))
                    max_branch_length = float(np.mean(max_branch_lengths))
                    weighted_mean = _interpolate_by_branch_length(
                        min_value,
                        max_value,
                        min_branch_length,
                        max_branch_length,
                    )
                codon_position_thetas[param] = weighted_mean
            root_thetas.append(codon_position_thetas)
    else:
        raise ValueError("model must contain either 'F1X4' or 'F3X4'")
    return root_thetas
