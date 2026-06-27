"""Cross-check the from-scratch CART tree against scikit-learn."""

import numpy as np
import pytest

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression

from statlearn import DecisionTree
from statlearn.metrics import accuracy_score, r2_score


def test_classification_tree_matches_sklearn():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=1
    )
    ours = DecisionTree(task="classification", max_depth=4).fit(X, y)
    sk = DecisionTreeClassifier(max_depth=4, criterion="gini", random_state=0).fit(X, y)
    acc_ours = accuracy_score(y, ours.predict(X))
    acc_sk = accuracy_score(y, sk.predict(X))
    assert abs(acc_ours - acc_sk) < 0.05


def test_regression_tree_matches_sklearn():
    X, y = make_regression(n_samples=200, n_features=4, noise=5.0, random_state=2)
    ours = DecisionTree(task="regression", max_depth=4).fit(X, y)
    sk = DecisionTreeRegressor(max_depth=4, criterion="squared_error", random_state=0).fit(X, y)
    r2_ours = r2_score(y, ours.predict(X))
    r2_sk = r2_score(y, sk.predict(X))
    assert abs(r2_ours - r2_sk) < 0.05


def test_tree_perfectly_fits_separable_data():
    # A clean axis-aligned split should be learned exactly.
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTree(task="classification").fit(X, y)
    assert accuracy_score(y, tree.predict(X)) == 1.0


def test_depth_zero_is_a_constant_predictor():
    X, y = make_classification(
        n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=3
    )
    tree = DecisionTree(task="classification", max_depth=0).fit(X, y)
    preds = tree.predict(X)
    # With no splits allowed, every prediction is the majority class.
    assert len(np.unique(preds)) == 1
