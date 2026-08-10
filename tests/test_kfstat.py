import unittest

import matplotlib
import numpy as np
import scipy.stats as stats

matplotlib.use("Agg")

from kftools import kfstat


class TestKFStat(unittest.TestCase):
    def test_kfstat(self):
        x = np.random.normal(size=200)
        y = np.random.normal(loc=0.2, size=200)
        out = kfstat.bm_test(x, y)
        self.assertEqual(len(out), 6)

    def test_kfstat_input_validation(self):
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.bm_test([1], [2, 3])
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.brunner_munzel_test([1], [2])
        with self.assertRaisesRegex(ValueError, "ttype must be a finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], ttype="bad")
        with self.assertRaisesRegex(ValueError, "ttype must be a finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], ttype=np.nan)
        with self.assertRaisesRegex(ValueError, "ttype must be a finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], ttype=np.inf)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfstat.bm_test([1, 2], [2, 3], alpha=0)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfstat.bm_test([1, 2], [2, 3], alpha=1)
        with self.assertRaisesRegex(ValueError, "finite numeric value"):
            kfstat.bm_test([1, 2], [2, 3], alpha=np.nan)
        with self.assertRaisesRegex(ValueError, "one of"):
            kfstat.brunner_munzel_test([1, 2], [2, 3], alternative="bad")
        out_two_sided_dash = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="two-sided")
        out_two_sided_dot = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="two.sided")
        out_two_sided_space = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="two sided")
        out_two_sided_upper = kfstat.brunner_munzel_test([1, 2, 3], [2, 3, 4], alternative="TWO_SIDED")
        self.assertAlmostEqual(out_two_sided_dash[1], out_two_sided_dot[1])
        self.assertAlmostEqual(out_two_sided_dash[1], out_two_sided_space[1])
        self.assertAlmostEqual(out_two_sided_dash[1], out_two_sided_upper[1])
        with self.assertRaisesRegex(ValueError, "must be a string"):
            kfstat.brunner_munzel_test([1, 2], [2, 3], alternative=None)
        with self.assertRaisesRegex(ValueError, "must contain numeric values"):
            kfstat.bm_test([1, {}], [2, 3])
        with self.assertRaisesRegex(ValueError, "must contain numeric values"):
            kfstat.brunner_munzel_test([1, {}], [2, 3])
        x_with_nan = [0.0, 1.0, np.nan, 2.0]
        y_with_inf = [1.0, 2.0, np.inf, 3.0]
        stat_filtered, p_filtered = kfstat.brunner_munzel_test(x_with_nan, y_with_inf, alternative="two_sided")
        stat_ref = stats.brunnermunzel([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], alternative="two-sided").statistic
        p_ref = stats.brunnermunzel([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], alternative="two-sided").pvalue
        self.assertAlmostEqual(stat_filtered, stat_ref)
        self.assertAlmostEqual(p_filtered, p_ref)
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.brunner_munzel_test([np.nan, np.inf], [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "at least 2 values"):
            kfstat.brunner_munzel_test([1.0, 2.0, 3.0], [np.nan, np.inf])
        with self.assertRaisesRegex(ValueError, "pooled variance is zero"):
            kfstat.bm_test([1, 1], [2, 2])
        with self.assertRaisesRegex(ValueError, "pooled variance is zero"):
            kfstat.brunner_munzel_test([1, 1], [2, 2])
        rng = np.random.default_rng(0)
        x = rng.normal(0.0, 1.0, 200)
        y = rng.normal(1.0, 1.0, 200)
        bm_two_sided = kfstat.bm_test(x, y)[2]
        bm_less = kfstat.bm_test(x, y, ttype=1)[2]
        bm_greater = kfstat.bm_test(x, y, ttype=-1)[2]
        bm_two_sided_ref = stats.brunnermunzel(x, y, alternative="two-sided").pvalue
        bm_less_ref = stats.brunnermunzel(x, y, alternative="less").pvalue
        bm_greater_ref = stats.brunnermunzel(x, y, alternative="greater").pvalue
        self.assertGreater(bm_two_sided, 0.0)
        self.assertGreater(bm_less, 0.0)
        self.assertAlmostEqual(bm_two_sided, bm_two_sided_ref)
        self.assertAlmostEqual(bm_less, bm_less_ref)
        self.assertAlmostEqual(bm_greater, bm_greater_ref)
        bm2_two_sided = kfstat.brunner_munzel_test(x, y, alternative="two_sided")[1]
        bm2_less = kfstat.brunner_munzel_test(x, y, alternative="less")[1]
        self.assertGreater(bm2_two_sided, 0.0)
        self.assertGreater(bm2_less, 0.0)
        self.assertAlmostEqual(bm2_two_sided, bm_two_sided_ref)
        self.assertAlmostEqual(bm2_less, bm_less_ref)
