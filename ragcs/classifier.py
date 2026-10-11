"""Logistic-regression head (Eqs. 9-10). Trains in seconds on CPU."""
from __future__ import annotations

from typing import Optional, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .features import FEATURE_NAMES


class HallucinationClassifier:
    def __init__(
        self,
        C: float = 1.0,
        seed: int = 42,
        feature_names: Sequence[str] = FEATURE_NAMES,
    ):
        self.C = C
        self.seed = seed
        self.feature_names = tuple(feature_names)
        self.model: Optional[LogisticRegression] = None
        self.cv_results_: Optional[dict] = None

    def _new_model(self, C: float) -> LogisticRegression:
        # L2 penalty is the sklearn default.
        return LogisticRegression(C=C, max_iter=1000, random_state=self.seed)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune: bool = True,
        cv: int = 5,
        c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    ) -> "HallucinationClassifier":
        """Fit the classifier; with ``tune`` select C by 5-fold stratified CV (F1)."""
        if tune:
            search = GridSearchCV(
                self._new_model(1.0),
                {"C": list(c_grid)},
                scoring="f1",
                cv=StratifiedKFold(cv, shuffle=True, random_state=self.seed),
            )
            search.fit(X, y)
            self.C = float(search.best_params_["C"])
            self.cv_results_ = {"best_C": self.C, "best_cv_f1": float(search.best_score_)}
        self.model = self._new_model(self.C).fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """p(y=1 | f), Eq. 9."""
        return self.model.predict_proba(np.atleast_2d(X))[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Eq. 10."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str) -> None:
        joblib.dump(
            {"model": self.model, "C": self.C, "seed": self.seed,
             "feature_names": self.feature_names, "cv_results": self.cv_results_},
            path,
        )

    @classmethod
    def load(cls, path: str) -> "HallucinationClassifier":
        blob = joblib.load(path)
        obj = cls(C=blob["C"], seed=blob["seed"], feature_names=blob["feature_names"])
        obj.model = blob["model"]
        obj.cv_results_ = blob.get("cv_results")
        return obj
