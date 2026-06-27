"""Cross-check the from-scratch logistic regression against scikit-learn."""

import numpy as np
import pytest

from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.datasets import make_classification

from statlearn import LogisticRegression
from statlearn.metrics import accuracy_score


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    return X, y


def test_logistic_accuracy_close_to_sklearn(clf_data):
    X, y = clf_data
    ours = LogisticRegression(lr=0.5, n_iters=20000, C=1e6).fit(X, y)
    sk = SkLogistic(C=1e6, max_iter=10000).fit(X, y)
    acc_ours = accuracy_score(y, ours.predict(X))
    acc_sk = accuracy_score(y, sk.predict(X))
    # Both fit the same (near-unregularised) objective: accuracy should match.
    assert abs(acc_ours - acc_sk) < 0.02


def test_logistic_coefficients_close_to_sklearn(clf_data):
    X, y = clf_data
    ours = LogisticRegression(lr=0.5, n_iters=30000, C=1e6).fit(X, y)
    sk = SkLogistic(C=1e6, max_iter=10000).fit(X, y)
    # Direction of the decision boundary should agree closely.
    assert np.allclose(ours.coef_, sk.coef_.ravel(), atol=0.15)


def test_predict_proba_range(clf_data):
    X, y = clf_data
    model = LogisticRegression().fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (X.shape[0],)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_predict_labels_binary(clf_data):
    X, y = clf_data
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert model.score(X, y) > 0.8
