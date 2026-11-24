"""Regression checks for ETF direction scores and forward-only rolling fits."""
import unittest

import numpy as np
import pandas as pd

from rdagent.scenarios.qlib.experiment.rd_tabular import FactorEstimator


class SampleDataset:
    def __init__(self):
        rng = np.random.default_rng(2026)
        dates = pd.bdate_range("2025-01-02", periods=100)
        index = pd.MultiIndex.from_product([dates, ["SH588170"]], names=["datetime", "instrument"])
        x = pd.DataFrame(rng.normal(size=(100, 2)), index=index, columns=["a", "b"])
        self.rows = pd.concat({"feature": x, "label": x[["a"]] * .01}, axis=1)
        self.segments = {"train": slice(0, 50), "valid": slice(50, 75), "test": slice(75, 100)}

    def prepare(self, segment, col_set, data_key):
        if isinstance(segment, list):
            return [self.prepare(s, col_set, data_key) for s in segment]
        return self.rows.iloc[self.segments[segment]][col_set]


class FactorEstimatorTest(unittest.TestCase):
    def test_selectable_methods(self):
        data = SampleDataset()
        for method in ("group", "linear", "ridge", "elastic", "forest", "lightgbm", "logistic"):
            with self.subTest(method=method):
                model = FactorEstimator(method)
                model.fit(data)
                out = model.predict(data)
                self.assertTrue(out.index.equals(data.rows.index[75:]))
                self.assertTrue(np.isfinite(out).all())

    def test_walk_forward_excludes_unknown_labels(self):
        for strategy in ("rolling", "expanding"):
            with self.subTest(strategy=strategy):
                data = SampleDataset()
                model = FactorEstimator("ridge", strategy, horizon=3, refit_days=5, window_days=25)
                model.fit(data)
                original = model.predict(data)
                # At test index 75 the labels of sessions 71..74 have not been used.
                data.rows.iloc[71:, -1] += .2
                changed = model.predict(data)
                pd.testing.assert_series_equal(original.iloc[:5], changed.iloc[:5])
                self.assertFalse(np.allclose(original.iloc[-5:], changed.iloc[-5:]))
