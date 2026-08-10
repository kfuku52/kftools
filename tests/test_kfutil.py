import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from kftools import kfutil


class TestKFUtil(unittest.TestCase):
    def test_kfutil(self):
        self.assertEqual(kfutil.add_dict_key_prefix({"a": 1}, "p"), {"p_a": 1})
        self.assertEqual(kfutil.add_dict_key_prefix({1: 2}, "p"), {"p_1": 2})
        with self.assertRaisesRegex(ValueError, "mapping type"):
            kfutil.add_dict_key_prefix([("a", 1)], "p")
        with self.assertRaisesRegex(ValueError, "prefix must be a string"):
            kfutil.add_dict_key_prefix({"a": 1}, 1)
        self.assertEqual(kfutil.rgb_to_hex(1, 0.5, 0), "#FF8000")
        self.assertEqual(len(kfutil.get_rgb_gradient(5, [1, 0, 0], [0, 0, 1])), 5)
        grad3 = kfutil.get_rgb_gradient(3, [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5])
        self.assertEqual(grad3[0], [1.0, 0.0, 0.0])
        self.assertEqual(grad3[1], [0.5, 0.5, 0.5])
        self.assertEqual(grad3[2], [0.0, 0.0, 1.0])
        grad2 = kfutil.get_rgb_gradient(2, [1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5])
        self.assertEqual(grad2[0], [1.0, 0.0, 0.0])
        self.assertEqual(grad2[1], [0.0, 0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "ncol must be an integer"):
            kfutil.get_rgb_gradient(2.5, [1, 0, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "exactly 3 channel values"):
            kfutil.get_rgb_gradient(3, [1, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "exactly 3 channel values"):
            kfutil.get_rgb_gradient(3, object(), [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfutil.get_rgb_gradient(3, [2, 0, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "bool is not allowed"):
            kfutil.get_rgb_gradient(3, [True, 0, 0], [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "bool is not allowed"):
            kfutil.get_rgb_gradient(3, [1, 0, 0], [0, False, 1])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfutil.rgb_to_hex(-0.1, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            kfutil.rgb_to_hex(1.1, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            kfutil.rgb_to_hex("a", 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            kfutil.rgb_to_hex(np.nan, 0.0, 0.0)
