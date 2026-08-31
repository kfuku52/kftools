import numbers
import re
from collections.abc import Mapping, Sequence

import numpy as np
from ete4 import PhyloTree

from ._typing import PathInput
from ._validation import coerce_path_argument

NUCLEOTIDES = ("A", "T", "C", "G")
CODON_POSITIONS = (0, 1, 2)
CODON_LENGTH = 3


def _validate_model_string(model):
    if not isinstance(model, str):
        raise ValueError("model must be a string")
    if model.strip() == "":
        raise ValueError("model must not be empty")


def _frequency_model_kind(model):
    has_f1x4 = "F1X4" in model
    has_f3x4 = "F3X4" in model
    if has_f1x4 == has_f3x4:
        raise ValueError("model must contain exactly one of 'F1X4' or 'F3X4'")
    return "F1X4" if has_f1x4 else "F3X4"


def _normalize_nucleotide_frequencies(freqs):
    scale_factor = sum(freqs.values())
    if (not np.isfinite(scale_factor)) or (scale_factor <= 0):
        raise ValueError("Nucleotide frequencies must have a positive total")
    for nuc in freqs:
        freqs[nuc] = freqs[nuc] / scale_factor


def _canonicalize_codon_frequencies(codon_freqs):
    canonicalized: dict[str, float] = {}
    for codon, codon_freq in codon_freqs.items():
        if (not isinstance(codon, str)) or (len(codon) != CODON_LENGTH):
            raise ValueError("codon_freqs keys must be codon strings of length 3")
        if isinstance(codon_freq, bool) or (not isinstance(codon_freq, numbers.Real)):
            raise ValueError("codon frequencies must be finite numeric values")
        codon_freq = float(codon_freq)
        if not np.isfinite(codon_freq):
            raise ValueError("codon frequencies must be finite numeric values")
        if codon_freq < 0:
            raise ValueError("codon frequencies must be non-negative")
        codon_upper = codon.upper()
        invalid_nucleotides = sorted(set(codon_upper) - set(NUCLEOTIDES))
        if len(invalid_nucleotides) > 0:
            raise ValueError(f"codon '{codon}' contains invalid nucleotides: {invalid_nucleotides}")
        canonicalized[codon_upper] = canonicalized.get(codon_upper, 0) + codon_freq
    return canonicalized


def _validate_nucleotide_frequency_dict(freqs):
    missing_nucleotides = sorted(set(NUCLEOTIDES) - set(freqs.keys()))
    if len(missing_nucleotides) > 0:
        raise ValueError(f"nucleotide frequency dictionary is missing keys: {missing_nucleotides}")
    unknown_nucleotides = set(freqs) - set(NUCLEOTIDES)
    if unknown_nucleotides:
        raise ValueError(f"nucleotide frequency dictionary has unsupported keys: {list(unknown_nucleotides)}")
    for nuc in NUCLEOTIDES:
        value = freqs[nuc]
        if isinstance(value, bool) or (not isinstance(value, numbers.Real)):
            raise ValueError(f"nucleotide frequency for '{nuc}' must be a finite numeric value")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"nucleotide frequency for '{nuc}' must be a finite numeric value")
        if value < 0:
            raise ValueError(f"nucleotide frequency for '{nuc}' must be non-negative")


def codon2nuc_freqs(
    codon_freqs: Mapping[str, float] | None = None,
    model: str = "",
) -> list[dict[str, float]]:
    """Convert codon frequencies to F1X4 or F3X4 nucleotide frequencies."""
    if codon_freqs is None:
        codon_freqs = {}
    if not isinstance(codon_freqs, Mapping):
        raise ValueError("codon_freqs must be a mapping from codon strings to frequencies")
    _validate_model_string(model)
    frequency_model = _frequency_model_kind(model)
    codon_freqs = _canonicalize_codon_frequencies(codon_freqs)
    if frequency_model == "F1X4":
        nuc_freqs = [dict.fromkeys(NUCLEOTIDES, 0.0)]
        for codon, codon_freq in codon_freqs.items():
            for nuc in NUCLEOTIDES:
                nuc_count = sum(nuc == c for c in codon)
                nuc_freqs[0][nuc] += codon_freq * nuc_count / CODON_LENGTH
    else:
        nuc_freqs = [dict.fromkeys(NUCLEOTIDES, 0.0) for _ in CODON_POSITIONS]
        for codon_pos in CODON_POSITIONS:
            for codon, codon_freq in codon_freqs.items():
                nuc_freqs[codon_pos][codon[codon_pos]] += codon_freq / CODON_LENGTH
    for freqs in nuc_freqs:
        _normalize_nucleotide_frequencies(freqs)
    return nuc_freqs


def nuc_freq2theta(nuc_freqs: Sequence[dict[str, float]] | None = None) -> list[dict[str, float]]:
    """Normalize A/T/C/G dictionaries and return mapNH theta parameters.

    Each input dictionary must have exactly those four keys and finite,
    non-negative values with a positive total. Inputs are not modified.
    Zero AT or GC totals give theta1 or theta2 of 0.5, respectively.
    """
    if nuc_freqs is None:
        nuc_freqs = []
    if not isinstance(nuc_freqs, (list, tuple)):
        raise ValueError("nuc_freqs must be a list or tuple of nucleotide-frequency dictionaries")
    thetas = []
    for freqs in nuc_freqs:
        if not isinstance(freqs, dict):
            raise ValueError("each entry in nuc_freqs must be a dictionary")
        _validate_nucleotide_frequency_dict(freqs)
        freqs = dict(freqs)
        _normalize_nucleotide_frequencies(freqs)
        AT_freq = freqs["A"] + freqs["T"]
        GC_freq = freqs["G"] + freqs["C"]
        theta = GC_freq
        theta1 = freqs["A"] / (freqs["A"] + freqs["T"]) if AT_freq != 0 else 0.5
        theta2 = freqs["G"] / (freqs["G"] + freqs["C"]) if GC_freq != 0 else 0.5
        thetas.append({"theta": theta, "theta1": theta1, "theta2": theta2})
    return thetas


def _validate_theta_entries(thetas):
    required_theta_keys = {"theta", "theta1", "theta2"}
    for theta in thetas:
        if not isinstance(theta, dict):
            raise ValueError("each theta entry must be a dictionary")
        missing_keys = sorted(required_theta_keys - set(theta.keys()))
        if len(missing_keys) > 0:
            raise ValueError(f"theta entry is missing keys: {missing_keys}")
        for theta_key in required_theta_keys:
            theta_value = theta[theta_key]
            if isinstance(theta_value, bool) or (not isinstance(theta_value, numbers.Real)):
                raise ValueError(f"theta entry key '{theta_key}' must be a finite numeric value")
            theta_value = float(theta_value)
            if not np.isfinite(theta_value):
                raise ValueError(f"theta entry key '{theta_key}' must be a finite numeric value")
            if (theta_value < 0) or (theta_value > 1):
                raise ValueError(f"theta entry key '{theta_key}' must be between 0 and 1")


def get_mapnh_thetas(model: str, thetas: Sequence[dict[str, float]] | None) -> str:
    """Render validated F1X4 or F3X4 theta values as a mapNH model string."""
    _validate_model_string(model)
    frequency_model = _frequency_model_kind(model)
    if thetas is None:
        thetas = []
    if not isinstance(thetas, (list, tuple)):
        raise ValueError("thetas must be a list or tuple of theta dictionaries")
    _validate_theta_entries(thetas)
    model_frequency = model
    model_frequency = re.sub(r"X4\+.*", "X4", model_frequency)
    model_frequency = re.sub(r".*\+F", "F", model_frequency)
    expected_count = 1 if frequency_model == "F1X4" else len(CODON_POSITIONS)
    if len(thetas) not in (0, expected_count):
        raise ValueError(f"{frequency_model} requires either 0 or {expected_count} theta entries; got {len(thetas)}")
    values = []
    for i, theta in enumerate(thetas):
        if len(thetas) == 1:
            values.append("Full.theta=" + str(theta["theta"]))
            values.append("Full.theta1=" + str(theta["theta1"]))
            values.append("Full.theta2=" + str(theta["theta2"]))
        else:
            values.append(str(i + 1) + "_Full.theta=" + str(theta["theta"]))
            values.append(str(i + 1) + "_Full.theta1=" + str(theta["theta1"]))
            values.append(str(i + 1) + "_Full.theta2=" + str(theta["theta2"]))
    if len(values) == 0:
        return model_frequency + "()"
    return model_frequency + "(" + ",".join(values) + ")"


def _read_target_fasta_sequence(alignment_file, leaf_name):
    seq_chunks: list[str] = []
    in_target = False
    found_target = False
    target_header_count = 0
    try:
        with open(alignment_file) as f:
            for line in f:
                if line.startswith(">"):
                    header = line[1:].strip()
                    header_id = header.split()[0] if header else ""
                    is_target_header = (header == leaf_name) or (header_id == leaf_name)
                    if is_target_header:
                        target_header_count += 1
                        if target_header_count > 1:
                            raise ValueError(f"leaf_name '{leaf_name}' appears multiple times in alignment_file")
                        found_target = True
                        in_target = True
                        seq_chunks = []
                    else:
                        in_target = False
                elif in_target:
                    seq_chunks.append(line.strip())
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read alignment_file: {alignment_file}") from exc
    seq = "".join(seq_chunks) if seq_chunks else None
    if found_target and (seq is None):
        raise ValueError(f"Sequence for leaf '{leaf_name}' is empty in alignment_file")
    if seq is None:
        raise ValueError(f"leaf_name '{leaf_name}' was not found in alignment_file")
    return seq.upper()


def _f3x4_nucleotide_frequencies(seq, leaf_name):
    invalid_nucleotides = sorted(set(seq) - set(NUCLEOTIDES))
    if len(invalid_nucleotides) > 0:
        raise ValueError(f"Sequence for leaf '{leaf_name}' contains invalid nucleotides: {invalid_nucleotides}")
    if len(seq) < CODON_LENGTH:
        raise ValueError("F3X4 requires sequences with at least three nucleotides")
    if len(seq) % CODON_LENGTH != 0:
        raise ValueError("F3X4 requires sequence lengths to be a multiple of three")
    seq_codons = [seq[start::3] for start in CODON_POSITIONS]
    return [{nuc: codon_seq.count(nuc) / len(codon_seq) for nuc in NUCLEOTIDES} for codon_seq in seq_codons]


def alignment2nuc_freqs(leaf_name: str, alignment_file: PathInput, model: str) -> list[dict[str, float]]:
    """Calculate F3X4 frequencies for one sequence in a plain-text FASTA file.

    Match the full header or its first whitespace-separated identifier. The
    selected sequence must contain only A/T/C/G and complete codons; gaps and
    ambiguity codes are rejected. Other records' lengths are not checked.
    F1X4 raises NotImplementedError after input validation.
    """
    _validate_model_string(model)
    frequency_model = _frequency_model_kind(model)
    if (not isinstance(leaf_name, str)) or (leaf_name.strip() == ""):
        raise ValueError("leaf_name must be a non-empty string")
    alignment_file = coerce_path_argument(alignment_file, "alignment_file")
    seq = _read_target_fasta_sequence(alignment_file, leaf_name)
    if frequency_model == "F1X4":
        raise NotImplementedError("F1X4 is not yet implemented")
    return _f3x4_nucleotide_frequencies(seq, leaf_name)


def _validate_subroot_nodes(subroot_thetas, tree):
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
    subroot_branch_lengths: dict[str, float] = {}
    for subroot_node in subroot_nodes:
        if subroot_node.name not in subroot_thetas:
            raise ValueError(f"subroot_thetas is missing node '{subroot_node.name}'")
        subroot_branch_lengths[subroot_node.name] = _validate_subroot_branch_length(
            subroot_node.dist, subroot_node.name
        )
    return subroot_names, subroot_branch_lengths


def _validate_subroot_branch_length(branch_length, subroot_name):
    if isinstance(branch_length, bool) or (not isinstance(branch_length, numbers.Real)):
        raise ValueError(f"Branch length for subroot node '{subroot_name}' must be a finite numeric value")
    branch_length = float(branch_length)
    if not np.isfinite(branch_length):
        raise ValueError(f"Branch length for subroot node '{subroot_name}' must be a finite numeric value")
    if branch_length < 0:
        raise ValueError(f"Branch length for subroot node '{subroot_name}' must be non-negative")
    return branch_length


def _validate_theta_position_entry(theta_entry, subroot_name, codon_position):
    if not isinstance(theta_entry, dict):
        raise ValueError(f"subroot_thetas['{subroot_name}'][{codon_position}] must be a dictionary of theta parameters")
    for param_name, param_value in theta_entry.items():
        if isinstance(param_value, bool) or (not isinstance(param_value, numbers.Real)):
            raise ValueError(
                f"subroot_thetas['{subroot_name}'][{codon_position}]['{param_name}'] must be a finite numeric value"
            )
        param_value = float(param_value)
        if not np.isfinite(param_value):
            raise ValueError(
                f"subroot_thetas['{subroot_name}'][{codon_position}]['{param_name}'] must be a finite numeric value"
            )
        if (param_value < 0) or (param_value > 1):
            raise ValueError(
                f"subroot_thetas['{subroot_name}'][{codon_position}]['{param_name}'] must be between 0 and 1"
            )


def _validate_subroot_theta_entries(subroot_thetas, subroot_names):
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
            _validate_theta_position_entry(theta_entry, subroot_name, codon_position)
    reference_params = set(subroot_thetas[subroot_names[0]][0].keys())
    for subroot_name in subroot_names:
        for codon_position in CODON_POSITIONS:
            params_here = set(subroot_thetas[subroot_name][codon_position].keys())
            if params_here != reference_params:
                raise ValueError(
                    "All subroot theta dictionaries must share identical parameter keys across codon positions"
                )
    return reference_params


def _average_root_theta_positions(subroot_thetas, subroot_names, params, branch_lengths):
    zero_length = branch_lengths == 0
    weights = None if zero_length.any() else np.reciprocal(branch_lengths)
    root_thetas = []
    for codon_position in CODON_POSITIONS:
        codon_position_thetas = {}
        for param in params:
            values = np.asarray(
                [subroot_thetas[name][codon_position][param] for name in subroot_names],
                dtype=float,
            )
            if zero_length.any():
                weighted_mean = float(values[zero_length].mean())
            else:
                weighted_mean = float(np.average(values, weights=weights))
            codon_position_thetas[param] = weighted_mean
        root_thetas.append(codon_position_thetas)
    return root_thetas


def weighted_mean_root_thetas(
    subroot_thetas: dict[str, Sequence[dict[str, float]]],
    tree: PhyloTree,
    model: str,
) -> list[dict[str, float]]:
    """Estimate F3X4 root thetas using inverse root-child branch lengths.

    If any root-child branches have zero length, average only those children
    equally. Children need unique non-empty names and three parameter dicts
    with matching keys. F1X4 raises NotImplementedError after validation.
    """
    _validate_model_string(model)
    frequency_model = _frequency_model_kind(model)
    subroot_names, subroot_branch_lengths = _validate_subroot_nodes(subroot_thetas, tree)
    reference_params = _validate_subroot_theta_entries(subroot_thetas, subroot_names)
    if frequency_model == "F1X4":
        raise NotImplementedError("F1X4 is not yet implemented")

    params = list(reference_params)
    branch_lengths = np.asarray(
        [subroot_branch_lengths[subroot_name] for subroot_name in subroot_names],
        dtype=float,
    )
    return _average_root_theta_positions(subroot_thetas, subroot_names, params, branch_lengths)
