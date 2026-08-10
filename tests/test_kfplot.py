import unittest
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kftools import kfplot


class TestKFPlot(unittest.TestCase):
    def test_kfplot(self):
        df = pd.DataFrame({"v": np.random.normal(size=80), "g": ["A"] * 40 + ["B"] * 40})
        ax = kfplot.hist_boxplot(x="v", category="g", df=df, xlim=[-3, 3])
        self.assertIsNotNone(ax)
        import matplotlib.pyplot as plt

        df_bar = pd.DataFrame({"x1": [1, 2], "x2": [2, 3], "y": ["A", "B"]})
        fig2, ax2 = plt.subplots()
        out_ax2 = kfplot.stacked_barplot(x=["x1", "x2"], y="y", data=df_bar, colors=["C0", "C1"], ax=ax2)
        self.assertIsNotNone(out_ax2)
        colors = ["C0", "C1"]
        fig3, ax3 = plt.subplots()
        _ = kfplot.hist_boxplot(x="v", category="g", df=df, colors=colors, xlim=[-3, 3], ax=ax3)
        self.assertEqual(colors, ["C0", "C1"])
        plt.close(fig3)
        fig3_tuple, ax3_tuple = plt.subplots()
        df_single_cat = pd.DataFrame({"v": [1.0, 2.0, 3.0], "g": ["A", "A", "A"]})
        _ = kfplot.hist_boxplot(
            x="v",
            category="g",
            df=df_single_cat,
            colors=("red",),
            xlim=[0, 4],
            ax=ax3_tuple,
        )
        self.assertEqual(ax3_tuple.lines[0].get_color(), "red")
        plt.close(fig3_tuple)
        fig3b, ax3b = plt.subplots()
        _ = kfplot.hist_boxplot(
            x="v",
            category="g",
            df=df,
            colors={"A": "C0"},
            xlim=[-3, 3],
            ax=ax3b,
        )
        yticklabels = [tick.get_text() for tick in ax3b.get_yticklabels()]
        self.assertIn("A", yticklabels)
        self.assertIn("B", yticklabels)
        plt.close(fig3b)
        fig4, ax4 = plt.subplots()
        out_ax4 = kfplot.density_scatter(
            x=df["v"].values,
            y=(df["v"].values * 0.5) + np.random.normal(scale=0.1, size=df.shape[0]),
            ax=ax4,
            cbar=False,
            show_cor_p=True,
        )
        self.assertIsNotNone(out_ax4)
        plt.close(fig4)
        plt.close(fig2)
        plt.close(ax.figure)

    def test_kfplot_density_scatter_empty(self):
        import statsmodels.api as sm

        with self.assertRaises(ValueError):
            kfplot.density_scatter(x=[np.nan], y=[np.nan], cbar=False)
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfplot.density_scatter(x=[1, {}], y=[1, 2], cbar=False)
        with self.assertRaisesRegex(ValueError, "same shape"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2, 3], cbar=False)
        with self.assertRaisesRegex(ValueError, "must include columns"):
            kfplot.density_scatter(x="x", y="y", df=pd.DataFrame({"x": [1, 2]}), cbar=False)
        with self.assertRaisesRegex(ValueError, "DataFrame-like"):
            kfplot.density_scatter(x="x", y="y", df=[], cbar=False)
        with self.assertRaisesRegex(ValueError, "string column name"):
            kfplot.density_scatter(x=["x"], y="y", df=pd.DataFrame({"x": [1, 2], "y": [1, 2]}), cbar=False)
        with self.assertRaisesRegex(ValueError, "string column name"):
            kfplot.density_scatter(x="x", y=["y"], df=pd.DataFrame({"x": [1, 2], "y": [1, 2]}), cbar=False)
        with self.assertRaisesRegex(ValueError, "must contain numeric values"):
            kfplot.density_scatter(
                x="x",
                y="y",
                df=pd.DataFrame({"x": ["a", "b"], "y": [1, 2]}),
                cbar=False,
            )
        with self.assertRaisesRegex(ValueError, "cor must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], cor="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "diag must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], diag="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "hue_log must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], hue_log="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "show_cor_p must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], show_cor_p="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "return_ims must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], return_ims="False", cbar=False)
        with self.assertRaisesRegex(ValueError, "cbar must be a boolean value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], cbar="False")
        ax = kfplot.density_scatter(
            x="x",
            y="y",
            df=pd.DataFrame({"x": ["1.0", "inf"], "y": [1.0, 2.0]}),
            cbar=False,
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close(ax.figure)
        ax_same_col = kfplot.density_scatter(
            x="x",
            y="x",
            df=pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
            cbar=False,
        )
        self.assertIsNotNone(ax_same_col)
        matplotlib.pyplot.close(ax_same_col.figure)
        with self.assertRaisesRegex(ValueError, "num_bin must be a positive integer"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], num_bin=0, cbar=False)
        with self.assertRaisesRegex(ValueError, "num_bin must be a positive integer"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], num_bin=2.5, cbar=False)
        with self.assertRaisesRegex(ValueError, "vmin must be None or a finite numeric value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], vmin=np.nan, cbar=False)
        with self.assertRaisesRegex(ValueError, "vmax must be None or a finite numeric value"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], vmax=np.inf, cbar=False)
        with self.assertRaisesRegex(ValueError, "vmin must be less than or equal to vmax"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], vmin=2, vmax=1, cbar=False)
        with self.assertRaisesRegex(ValueError, "plot_range must"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range=[0, 1], cbar=False)
        with self.assertRaisesRegex(ValueError, "plot_range must"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range="bad", cbar=False)
        with self.assertRaisesRegex(ValueError, "xmin <= xmax"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range=[2, 1, 0, 1], cbar=False)
        with self.assertRaisesRegex(ValueError, "ymin <= ymax"):
            kfplot.density_scatter(x=[1, 2], y=[1, 2], plot_range=[0, 1, 2, 1], cbar=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with self.assertRaisesRegex(ValueError, "GLM fit failed"):
                kfplot.density_scatter(x=[1, 2, 3], y=[0, 0, 0], reg_family=sm.families.Poisson(), cbar=False)
        with self.assertRaisesRegex(ValueError, "reg_family must be a statsmodels family object"):
            kfplot.density_scatter(x=[1, 2, 3], y=[1, 2, 3], reg_family="bad", cbar=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(
                x=[1, 2, 3, 4],
                y=[1, 2, 3, 4],
                reg_family=sm.families.Poisson(),
                cbar=False,
            )
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("log link function was detected" in str(wi.message) for wi in w),
                "density_scatter should not emit runtime warnings for expected Poisson log-link handling",
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(
                x=[1, 2, 3, 4],
                y=[0, 1, 2, 3],
                reg_family=sm.families.Poisson(),
                cbar=False,
            )
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("divide by zero encountered in log" in str(wi.message) for wi in w),
                "density_scatter should suppress NumPy divide warnings during log-link transformation",
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(x=[1.0], y=[1.0], cbar=False)
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("identical low and high" in str(wi.message) for wi in w),
                "density_scatter should avoid singular-axis matplotlib warnings for single-point inputs",
            )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = kfplot.density_scatter(x=[1.0], y=[1.0], cbar=False, show_cor_p=False)
            self.assertIsNotNone(ax)
            self.assertFalse(
                any("Degrees of freedom <= 0" in str(wi.message) for wi in w),
                "density_scatter(show_cor_p=False) should avoid corrcoef runtime warnings for single-point inputs",
            )

    def test_kfplot_stacked_barplot_input_validation(self):
        import matplotlib.pyplot as plt

        df = pd.DataFrame({"a": [1], "b": [2]})
        df_bad_x = pd.DataFrame({"a": [1, 2], "c": ["x", "y"], "g": ["A", "B"]})
        df_bad_y = pd.DataFrame({"g": ["A", "B"], "y1": [1, 2], "y2": ["x", "y"]})
        df_bad_group = pd.DataFrame({"a": [1, 2], "g": [[1], [2]]})
        fig, ax = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "Exactly one of x and y"):
                kfplot.stacked_barplot(x="a", y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "DataFrame-like"):
                kfplot.stacked_barplot(x=["a"], y="b", data=None, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "must contain at least one"):
                kfplot.stacked_barplot(x=[], y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x column name must be a non-empty"):
                kfplot.stacked_barplot(x="", y=["y1"], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y column name must be a non-empty"):
                kfplot.stacked_barplot(x=["a"], y="", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x list must contain non-empty string"):
                kfplot.stacked_barplot(x=[{}], y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y list must contain non-empty string"):
                kfplot.stacked_barplot(x="g", y=[None], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x list must not contain duplicate"):
                kfplot.stacked_barplot(x=["a", "a"], y="b", data=df, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y list must not contain duplicate"):
                kfplot.stacked_barplot(x="g", y=["y1", "y1"], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "x columns must contain numeric values"):
                kfplot.stacked_barplot(x=["a", "c"], y="g", data=df_bad_x, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "y columns must contain numeric values"):
                kfplot.stacked_barplot(x="g", y=["y1", "y2"], data=df_bad_y, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "must contain hashable values"):
                kfplot.stacked_barplot(x=["a"], y="g", data=df_bad_group, colors=["C0"], ax=ax)
            with self.assertRaisesRegex(ValueError, "must contain scalar values"):
                kfplot.stacked_barplot(
                    x=["a"],
                    y="g",
                    data=pd.DataFrame({"a": [1, 2], "g": [("A",), ("B",)]}),
                    colors=["C0"],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must not contain complex values"):
                kfplot.stacked_barplot(
                    x="g",
                    y=["a"],
                    data=pd.DataFrame({"g": [1 + 2j, 3 + 4j], "a": [1, 2]}),
                    colors=["C0"],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must not mix bool and numeric non-bool values"):
                kfplot.stacked_barplot(
                    x=["a"],
                    y="g",
                    data=pd.DataFrame({"a": [1, 2], "g": [1, True]}),
                    colors=["C0"],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must not contain non-finite numeric values"):
                kfplot.stacked_barplot(
                    x="x",
                    y=["a"],
                    data=pd.DataFrame({"x": [0.1, np.inf], "a": [1, 2]}),
                    colors=["C0"],
                    ax=ax,
                )
            out_bool = kfplot.stacked_barplot(
                x="g",
                y=["y1", "y2"],
                data=pd.DataFrame({"g": [1], "y1": [True], "y2": [1]}),
                colors=["C0", "C1"],
                ax=None,
            )
            self.assertIsNotNone(out_bool)
            plt.close(out_bool.figure)
            out_datetime = kfplot.stacked_barplot(
                x="g",
                y=["y1", "y2"],
                data=pd.DataFrame(
                    {
                        "g": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
                        "y1": [1, 2],
                        "y2": [2, 3],
                    }
                ),
                colors=["C0", "C1"],
                ax=None,
            )
            self.assertIsNotNone(out_datetime)
            plt.close(out_datetime.figure)
            out_numpy_scalar_color = kfplot.stacked_barplot(
                x="g",
                y=["y1", "y2"],
                data=pd.DataFrame({"g": ["A", "B"], "y1": [1, 2], "y2": [2, 3]}),
                colors=np.array("red"),
                ax=None,
            )
            self.assertIsNotNone(out_numpy_scalar_color)
            plt.close(out_numpy_scalar_color.figure)
            with self.assertRaisesRegex(ValueError, "must not mix string and non-string"):
                fig_count_before = len(plt.get_fignums())
                kfplot.stacked_barplot(
                    x="g",
                    y=["y1", "y2"],
                    data=pd.DataFrame({"g": [1, "2"], "y1": [1, 2], "y2": [2, 3]}),
                    colors=["C0", "C1"],
                    ax=None,
                )
            self.assertEqual(
                len(plt.get_fignums()),
                fig_count_before,
                "stacked_barplot should close internally created figures when validation fails after figure creation",
            )
            out = kfplot.stacked_barplot(x=["a"], y="b", data=df, colors=["C0"], ax=None)
            self.assertIsNotNone(out)
            plt.close(out.figure)
        finally:
            plt.close(fig)

    def test_hist_boxplot_input_validation(self):
        with self.assertRaisesRegex(ValueError, "DataFrame-like"):
            kfplot.hist_boxplot(x="x", category="g", df=[])
        with self.assertRaisesRegex(ValueError, "x must be a non-empty string"):
            kfplot.hist_boxplot(x=["x"], category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "category must be a non-empty string"):
            kfplot.hist_boxplot(x="x", category=["g"], df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "x must be a non-empty string"):
            kfplot.hist_boxplot(x="", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "category must be a non-empty string"):
            kfplot.hist_boxplot(x="x", category="", df=pd.DataFrame({"x": [1.0], "g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "alpha must be a finite numeric value"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), alpha="bad")
        with self.assertRaisesRegex(ValueError, "alpha must be between 0 and 1"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), alpha=2)
        with self.assertRaisesRegex(ValueError, "box_step must be a positive finite numeric value"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), box_step="bad")
        with self.assertRaisesRegex(ValueError, "box_step must be a positive finite numeric value"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0], "g": ["A"]}), box_step=-0.1)
        with self.assertRaisesRegex(ValueError, "must contain hashable values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [[1], [2]]}),
            )
        with self.assertRaisesRegex(ValueError, "must contain scalar values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [("A",), ("B",)]}),
            )
        with self.assertRaisesRegex(ValueError, "must not contain complex values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [1 + 2j, 3 + 4j]}),
            )
        with self.assertRaisesRegex(ValueError, "must not mix bool and numeric non-bool values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [1, True]}),
            )
        with self.assertRaisesRegex(ValueError, "must not contain non-finite numeric values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [np.inf, 1]}),
            )
        with self.assertRaisesRegex(ValueError, "at least one non-NaN"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame(columns=["x", "g"]))
        with self.assertRaisesRegex(ValueError, "category column"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"x": [1.0]}))
        with self.assertRaisesRegex(ValueError, "x column"):
            kfplot.hist_boxplot(x="x", category="g", df=pd.DataFrame({"g": ["A"]}))
        with self.assertRaisesRegex(ValueError, "at least one non-NaN"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": [np.nan, np.nan]}),
            )

    def test_hist_boxplot_range_bin_and_color_validation(self):
        with self.assertRaisesRegex(ValueError, "xlim must"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                xlim=[0],
            )
        with self.assertRaisesRegex(ValueError, "xmin <= xmax"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                xlim=[2, 1],
            )
        with self.assertRaisesRegex(ValueError, "numeric values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": ["bad", "2"], "g": ["A", "B"]}),
            )
        with self.assertRaisesRegex(ValueError, "finite numeric values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, np.inf], "g": ["A", "B"]}),
            )
        with self.assertRaisesRegex(ValueError, "bins must"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins="bad",
            )
        with self.assertRaisesRegex(ValueError, "finite numeric bin-edge values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=["a", "b"],
            )
        with self.assertRaisesRegex(ValueError, "at least 2 bin-edge"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=[1],
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=[2, 1],
            )
        with self.assertRaisesRegex(ValueError, "finite numeric bin-edge values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                bins=[1, np.inf],
            )
        with self.assertRaisesRegex(ValueError, "colors contains categories"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, 2.0], "g": ["A", "B"]}),
                colors={"A": "C0", "Z": "C1"},
            )

    def test_hist_boxplot_supported_types_and_cleanup(self):
        import matplotlib.pyplot as plt

        fig_count_before = len(plt.get_fignums())
        with self.assertRaisesRegex(ValueError, "has no non-NaN values"):
            kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [1.0, np.nan], "g": ["A", "B"]}),
            )
        self.assertEqual(
            len(plt.get_fignums()),
            fig_count_before,
            "hist_boxplot should close internally created figures when validation fails after figure creation",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax_bool_x = kfplot.hist_boxplot(
                x="x",
                category="g",
                df=pd.DataFrame({"x": [True, False, True], "g": ["A", "A", "B"]}),
            )
            self.assertIsNotNone(ax_bool_x)
            self.assertFalse(
                any("Converting input from bool" in str(wi.message) for wi in w),
                "hist_boxplot should coerce boolean x values to float and avoid matplotlib bool histogram warnings",
            )
            plt.close(ax_bool_x.figure)
        ax_datetime_category = kfplot.hist_boxplot(
            x="x",
            category="g",
            df=pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0],
                    "g": [
                        pd.Timestamp("2020-01-01"),
                        pd.Timestamp("2020-01-01"),
                        pd.Timestamp("2020-01-02"),
                    ],
                }
            ),
        )
        self.assertIsNotNone(ax_datetime_category)
        plt.close(ax_datetime_category.figure)
        ax_numpy_scalar_color = kfplot.hist_boxplot(
            x="x",
            category="g",
            df=pd.DataFrame({"x": [1.0, 2.0, 3.0], "g": ["A", "A", "B"]}),
            colors=np.array("red"),
        )
        self.assertIsNotNone(ax_numpy_scalar_color)
        plt.close(ax_numpy_scalar_color.figure)

    def test_ols_annotations_input_validation(self):
        fig, ax = plt.subplots()
        try:
            out = kfplot.ols_annotations(
                x=[1, 2, 3],
                y=[2, 3, 4],
                ax=ax,
                method="ols",
            )
            self.assertIsNotNone(out)
            with self.assertRaisesRegex(ValueError, "at least 2 rows"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame(columns=["x", "y"]),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "DataFrame-like"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=[],
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "string column name"):
                kfplot.ols_annotations(
                    x=["x"],
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "string column name"):
                kfplot.ols_annotations(
                    x="x",
                    y=["y"],
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must include columns"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "must be either"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    method="bad",
                )
            with self.assertRaisesRegex(ValueError, "requires numeric x and y"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": ["a", "b"], "y": [3, 4]}),
                    ax=ax,
                )
            with self.assertRaisesRegex(ValueError, "stats must be a string or a sequence"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    stats=1,
                )
            with self.assertRaisesRegex(ValueError, "unsupported entries"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    stats=["N", "bad"],
                )
            with self.assertRaisesRegex(ValueError, "unsupported entries"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    stats=["N", 1],
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy=[],
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy=[0.1],
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy="bad",
                )
            with self.assertRaisesRegex(ValueError, "textxy must contain exactly"):
                kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                    ax=ax,
                    textxy=None,
                )
        finally:
            plt.close(fig)

    def test_ols_annotations_regular_inputs(self):
        fig, ax = plt.subplots()
        try:
            out_stats_str = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
                ax=ax,
                stats="N",
            )
            self.assertIsNotNone(out_stats_str)
            indexed_df = pd.DataFrame(
                {"x": [3.0, 1.0, 2.0], "y": [2.9, 1.1, 2.1]},
                index=[10, 30, 20],
            )
            out_idx = kfplot.ols_annotations(
                x="x",
                y="y",
                data=indexed_df,
                ax=ax,
                method="ols",
            )
            self.assertIsNotNone(out_idx)
        finally:
            plt.close(fig)

    def test_ols_annotations_degenerate_inputs(self):
        fig, ax = plt.subplots()
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_two_point = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [2, False], "y": [1, 0]}),
                    ax=ax,
                    method="ols",
                    stats=["N", "rsquared", "rsquared_p"],
                )
                self.assertIsNotNone(out_two_point)
                self.assertFalse(
                    any("divide by zero encountered in divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid rsquared divide-by-zero warnings when residual dof is 0",
                )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_quantreg_degenerate = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [0, 0], "y": [0, False]}),
                    ax=ax,
                    method="quantreg",
                    stats=["N", "rsquared", "rsquared_p"],
                )
                self.assertIsNotNone(out_quantreg_degenerate)
                self.assertFalse(
                    any("invalid value encountered in scalar divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid quantreg prsquared divide warnings for degenerate inputs",
                )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_quantreg_constant_y = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame(
                        {
                            "x": [
                                -0.11544881657289713,
                                -0.9731576166416964,
                                1.5521497178256616,
                                0.8980692396695097,
                                0.8121127598698327,
                                -0.15301988744981074,
                                1.995012141280733,
                                0.2387915074956506,
                                0.02822408070062835,
                                1.2000253792148359,
                                -0.5006654582078435,
                                0.5738987504794617,
                                -1.0060237130999417,
                                1.23486392210854,
                            ],
                            "y": [0.0] * 14,
                        }
                    ),
                    ax=ax,
                    method="quantreg",
                    stats=["N", "slope", "slope_p"],
                )
                self.assertIsNotNone(out_quantreg_constant_y)
                self.assertFalse(
                    any("divide by zero encountered in scalar divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid quantreg fit divide-by-zero warnings when y has no variation",
                )
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                out_ols_constant_y = kfplot.ols_annotations(
                    x="x",
                    y="y",
                    data=pd.DataFrame({"x": [2, False, True], "y": [1, True, 1]}),
                    ax=ax,
                    method="ols",
                    stats=["N", "rsquared", "rsquared_p"],
                )
                self.assertIsNotNone(out_ols_constant_y)
                self.assertFalse(
                    any("divide by zero encountered in scalar divide" in str(wi.message) for wi in w),
                    "ols_annotations should avoid OLS rsquared divide warnings when y has no variation",
                )
            out_quantreg_missing_slope = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [False, True], "y": [0, 1]}),
                ax=ax,
                method="quantreg",
                stats=["N", "rsquared_p"],
            )
            self.assertIsNotNone(out_quantreg_missing_slope)
            out_quantreg_missing_slope_requested = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [False, True], "y": [0, 1]}),
                ax=ax,
                method="quantreg",
                stats=["slope", "slope_p"],
            )
            self.assertIsNotNone(out_quantreg_missing_slope_requested)
            out_bool_y = kfplot.ols_annotations(
                x="x",
                y="y",
                data=pd.DataFrame({"x": [3, 3], "y": [False, False]}),
                ax=ax,
                method="ols",
                stats=["slope", "slope_p"],
            )
            self.assertIsNotNone(out_bool_y)
        finally:
            plt.close(fig)
