import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from kftools import kfexpression


class TestKFExpression(unittest.TestCase):
    def test_kfexpression(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
        self.assertAlmostEqual(kfexpression.calc_complementarity([1, 2], [1, 1]), 0.25)
        with self.assertRaisesRegex(ValueError, "same number of values"):
            kfexpression.calc_complementarity([1, 2, 3], [1])
        with self.assertRaisesRegex(ValueError, "at least one value"):
            kfexpression.calc_complementarity([], [1])
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfexpression.calc_complementarity([1, {}], [1, 2])
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfexpression.calc_complementarity([1, np.inf], [1, 2])
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfexpression.calc_complementarity([1, 2], [1, np.nan])
        with self.assertRaisesRegex(ValueError, "at least one"):
            kfexpression.calc_tau(df, [], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "not found"):
            kfexpression.calc_tau(df, ["z"], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfexpression.calc_tau(pd.DataFrame({"a": ["x"]}), ["a"])
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            kfexpression.calc_tau(df, ["a", "a"], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "non-empty string column names"):
            kfexpression.calc_tau(df, [{}], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "non-empty string column names"):
            kfexpression.calc_tau(df, [""], unlog2=False, unPlus1=False)
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfexpression.calc_tau(
                pd.DataFrame({"a": [1.0, np.inf], "b": [2.0, 3.0]}),
                ["a", "b"],
                unlog2=False,
                unPlus1=False,
            )
        tau_zero = kfexpression.calc_tau(
            pd.DataFrame({"a": [0.0, 0.0]}),
            ["a"],
            unlog2=False,
            unPlus1=False,
        )
        self.assertEqual(tau_zero.tolist(), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "non-empty sequence"):
            kfexpression.calc_tau(df, 0)
        with self.assertRaisesRegex(ValueError, "DataFrame-like"):
            kfexpression.calc_tau(None, ["a"])
        with self.assertRaisesRegex(ValueError, "unlog2 must be a boolean value"):
            kfexpression.calc_tau(df, ["a", "b"], unlog2="False", unPlus1=False)
        with self.assertRaisesRegex(ValueError, "unPlus1 must be a boolean value"):
            kfexpression.calc_tau(df, ["a", "b"], unlog2=True, unPlus1="False")
        with self.assertRaisesRegex(ValueError, "out of range"):
            kfexpression.calc_tau(
                pd.DataFrame({"a": [2000.0], "b": [2000.0]}),
                ["a", "b"],
                unlog2=True,
                unPlus1=True,
            )
