"""Frequency normalization must preserve ratios across representable scales."""

import numpy as np
import pytest

from kftools import kfseq


@pytest.mark.parametrize("model", ["F1X4", "F3X4"])
@pytest.mark.parametrize("scale", [5e-324, 1.0, 1e308])
def test_codon_frequencies_preserve_scale_and_case_aliases(model, scale):
    frequencies = {"AAA": scale, "aaa": scale, "TTT": scale}
    before = frequencies.copy()
    result = kfseq.codon2nuc_freqs(frequencies, model)
    for position in result:
        assert position == pytest.approx({"A": 2 / 3, "T": 1 / 3, "C": 0, "G": 0})
    assert frequencies == before


@pytest.mark.parametrize("scale", [5e-324, 1.0, 1e308, np.int64(2**62)])
def test_nucleotide_frequencies_preserve_scale(scale):
    frequencies = dict.fromkeys("ATCG", scale)
    before = frequencies.copy()
    assert kfseq.nuc_freq2theta([frequencies]) == [{"theta": 0.5, "theta1": 0.5, "theta2": 0.5}]
    assert frequencies == before


@pytest.mark.parametrize("model", ["F3X4+F3X4", "F1X4+F1X4", "NOTF3X4", "F3X40", "F1X4+F3X4"])
def test_frequency_model_requires_one_complete_token(model):
    with pytest.raises(ValueError, match="exactly one"):
        kfseq.codon2nuc_freqs({"AAA": 1}, model)
    with pytest.raises(ValueError, match="exactly one"):
        kfseq.get_mapnh_thetas(model, [])
