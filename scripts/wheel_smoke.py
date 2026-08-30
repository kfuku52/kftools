"""Exercise the installed distribution from outside the checkout."""

import importlib
from importlib import metadata, resources
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kftools
from kftools import kfexpression, kfphylo, kfplot, kfseq, kfspecies

checkout = Path(__file__).resolve().parents[1]
assert not Path(kftools.__file__).resolve().is_relative_to(checkout), "imported source instead of the installed wheel"
assert metadata.version("kftools") == kftools.__version__
assert resources.files(kftools).joinpath("py.typed").is_file()
for name in ("kfexpression", "kfog", "kfphylo", "kfplot", "kfseq", "kfspecies", "kfstat", "kfutil"):
    importlib.import_module(f"kftools.{name}")

np.testing.assert_allclose(kfexpression.calc_tau(pd.DataFrame({"a": [10.0], "b": [0.0]}), ["a", "b"], False), [1])
assert kfphylo.get_tree_height("((A:1,B:1):2,C:3);") == 3
assert kfseq.nuc_freq2theta([dict.fromkeys("ATCG", 1.0)])[0]["theta"] == 0.5
assert kfspecies.parse_species_label("Quercus cf. robur", "taxonomic").species_label == "Quercus_cf_robur"
axis = kfplot.ols_annotations([1, 2, 3], [2, 4, 6], method="ols", stats="slope")
assert axis.texts[0].get_text().strip() == "slope = 2.00"
plt.close(axis.figure)
print(f"Installed wheel {kftools.__version__}: imports, py.typed, numerical APIs, and plotting passed")
