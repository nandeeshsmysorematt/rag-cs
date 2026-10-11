import numpy as np

from ragcs.classifier import HallucinationClassifier


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 4)) + np.where(y[:, None] == 1, -1.0, 1.0)
    return X, y


def test_fit_predict_and_roundtrip(tmp_path):
    X, y = _toy()
    clf = HallucinationClassifier().fit(X, y, tune=True)
    assert clf.cv_results_["best_C"] in (0.01, 0.1, 1.0, 10.0, 100.0)
    assert (clf.predict(X) == y).mean() > 0.85
    path = tmp_path / "clf.joblib"
    clf.save(str(path))
    clf2 = HallucinationClassifier.load(str(path))
    assert np.allclose(clf.predict_proba(X), clf2.predict_proba(X))
