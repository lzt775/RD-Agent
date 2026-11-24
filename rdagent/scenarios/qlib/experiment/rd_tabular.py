"""Selectable factor estimators for Qlib DatasetH, including purged walk-forward refits."""
import numpy as np
import pandas as pd
from qlib.model.base import Model
from qlib.data.dataset.handler import DataHandlerLP
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class FactorEstimator(Model):
    def __init__(self, method, validation="holdout", refit_days=20, window_days=120, horizon=1, tuning=None):
        self.method = method
        self.validation = validation
        self.refit_days = refit_days
        self.window_days = window_days
        self.horizon = horizon
        self.tuning = tuning or {}

    @staticmethod
    def _features(frame):
        return frame.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)

    def _fit(self, rows):
        rows = rows.loc[np.isfinite(rows["label"].iloc[:, 0])]
        if rows.empty:
            raise ValueError("The training window contains no available return labels")
        x, y = self._features(rows["feature"]), rows["label"].iloc[:, 0].to_numpy()
        if self.method == "group":
            # Choose one feature on training data only, then estimate returns in five training quantiles.
            xy = pd.DataFrame(x)
            corr = xy.corrwith(pd.Series(y), method="spearman").abs().fillna(0)
            self.col = int(corr.idxmax())
            groups = min(len(y), self.tuning.get("groups", 5))
            self.edges = np.unique(np.quantile(x[:, self.col], np.arange(1, groups) / groups))
            bins = np.searchsorted(self.edges, x[:, self.col])
            self.means = [y[bins == i].mean() if (bins == i).any() else y.mean() for i in range(len(self.edges)+1)]
            return
        if self.method == "logistic":
            up = y > 0
            self.class_means = None
            if up.all() or not up.any():
                self.model = DummyRegressor().fit(x, y)
                return
            self.class_means = [y[~up].mean(), y[up].mean()]
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, **self.tuning)).fit(x, up)
            return
        if self.method == "lightgbm":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=100, num_leaves=15, n_jobs=2, random_state=42, verbosity=-1)
        else:
            models = {"linear": LinearRegression(), "ridge": Ridge(alpha=1),
                      "elastic": ElasticNet(alpha=0.0001, max_iter=3000),
                      "forest": RandomForestRegressor(n_estimators=80, min_samples_leaf=5, random_state=42, n_jobs=2)}
            model = models[self.method]
        model.set_params(**self.tuning)
        self.model = make_pipeline(StandardScaler(), model).fit(x, y)

    def _score(self, frame):
        x = self._features(frame)
        if self.method == "group":
            values = np.take(self.means, np.searchsorted(self.edges, x[:, self.col]))
        elif self.method == "logistic" and self.class_means is not None:
            values = self.model.predict_proba(x) @ self.class_means
        else:
            values = self.model.predict(x)
        return pd.Series(values, index=frame.index)

    def fit(self, dataset, **kwargs):
        self._fit(dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L))

    def predict(self, dataset, segment="test"):
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if self.validation == "holdout" or segment != "test":
            return self._score(features)
        # Filter by session before each fit. The last label requires the open H+1 days later.
        rows = pd.concat(dataset.prepare(["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L))
        dates = rows.index.get_level_values("datetime")
        days = dates.unique().sort_values()
        predict_dates = features.index.get_level_values("datetime")
        test_days = predict_dates.unique().sort_values()
        predictions = []
        for offset in range(0, len(test_days), self.refit_days):
            block = test_days[offset:offset+self.refit_days]
            known = days[days < block[0]][:-(self.horizon + 1)]
            if self.validation == "rolling":
                known = known[-self.window_days:]
            self._fit(rows.loc[dates.isin(known)])
            predictions.append(self._score(features.loc[predict_dates.isin(block)]))
        return pd.concat(predictions).reindex(features.index)
