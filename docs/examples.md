# Usage examples

Calculate the standard tissue-specificity tau index:

```python
import pandas as pd

from kftools.kfexpression import calc_tau

expression = pd.DataFrame(
    {
        "leaf": [10.0, 5.0],
        "root": [0.0, 5.0],
    }
)
tau = calc_tau(expression, ["leaf", "root"], unlog2=False, unPlus1=False)
# array([1., 0.])
```

Read and inspect a phylogenetic tree:

```python
from kftools.kfphylo import check_ultrametric, get_tree_height

tree = "((A:1,B:1):2,C:3);"
assert check_ultrametric(tree)
assert get_tree_height(tree) == 3.0
```

Parse a taxonomic label:

```python
from kftools.kfspecies import parse_species_label

species = parse_species_label(
    "Bacillus_subtilis_subsp_168_gene3",
    species_parser="taxonomic",
)
assert species.scientific_name == "Bacillus subtilis subsp. 168"
```

Prepare a lookup once when many nearest-ancestor queries use the same table:

```python
import pandas as pd

from kftools.kfog import prepare_most_recent_lookup

branch_table = pd.DataFrame(
    {
        "orthogroup": ["og1", "og1"],
        "branch_id": [0, 1],
        "parent": [1, 1],
        "is_shift": [0, 1],
        "regime": [0, 1],
    }
)
lookup = prepare_most_recent_lookup(
    branch_table,
    target_col="is_shift",
    return_col="regime",
)
regime = lookup.find(0, "og1", target_value=1)
assert regime == 1
```

![kfplot examples](images/kfplot_examples.png)
