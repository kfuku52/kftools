"""Static consumer checks: valid types and deliberately rejected inputs.

Checked by `make typecheck`, not executed by pytest. Unused-ignore errors catch
regressions that would silently turn these public contracts back into Any.
"""

from typing import assert_type

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from kftools import kfexpression, kfog, kfplot, kfseq, kfspecies, kfutil

frame = pd.DataFrame({"leaf": [1.0], "root": [0.0]})
assert_type(kfexpression.calc_tau(frame, ["leaf", "root"]), NDArray[np.float64])
assert_type(kfplot.ols_annotations([1, 2], [2, 3], method="ols"), Axes)
assert_type(kfog.compute_delta(frame, "value"), pd.DataFrame)
assert_type(kfseq.nuc_freq2theta([{"A": 1, "T": 1, "C": 1, "G": 1}]), list[dict[str, float]])
assert_type(kfutil.add_dict_key_prefix({1: 2.0}, "prefix"), dict[str, float])
config: kfspecies.ParserConfig = {"type": "regex", "pattern": r"(\w+)_(\w+)", "group": [1, 2]}
assert_type(kfspecies.parse_species_label("A_b", config), kfspecies.SpeciesParseResult)

kfexpression.calc_tau([1, 2], ["leaf"])  # type: ignore[arg-type]
kfog.compute_delta("table.tsv", "value")  # type: ignore[arg-type]
kfseq.nuc_freq2theta([{"A": "not a number"}])  # type: ignore[dict-item]
kfplot.ols_annotations([1, 2], [2, 3], ax="not an axis")  # type: ignore[arg-type]
kfspecies.parse_species_label("A_b", species_parser=123)  # type: ignore[arg-type]
