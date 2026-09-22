import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from kftools import kfexpression


class TestKFExpression(unittest.TestCase):
    def test_tau_accepts_numpy_boolean_flags_without_mutating_input(self):
        df = pd.DataFrame({"a": [0.0, 1.0, 3.0], "b": [0.0, 2.0, 1.0]})
        original = df.copy(deep=True)
        for unlog2 in (False, True):
            for unPlus1 in (False, True):
                with self.subTest(unlog2=unlog2, unPlus1=unPlus1):
                    expected = kfexpression.calc_tau(df, ["a", "b"], unlog2, unPlus1)
                    actual = kfexpression.calc_tau(df, ["a", "b"], np.bool_(unlog2), np.bool_(unPlus1))
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(actual.dtype, expected.dtype)
        pd.testing.assert_frame_equal(df, original)

    def test_tau_validates_flags_before_columns(self):
        for unlog2, unPlus1, invalid_flag in [(0, 0, "unlog2"), (False, 0, "unPlus1")]:
            with self.subTest(invalid_flag=invalid_flag):
                with self.assertRaises(ValueError) as error:
                    kfexpression.calc_tau(None, [], unlog2, unPlus1)
                self.assertEqual(str(error.exception), f"{invalid_flag} must be a boolean value")

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
