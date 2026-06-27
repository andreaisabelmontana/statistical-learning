"""CART decision tree for regression and classification (numpy, from scratch).

Greedy recursive binary splits. Classification splits minimise Gini impurity;
regression splits minimise weighted variance (equivalently squared error).
Splits are chosen by an exhaustive scan over features and candidate thresholds.
"""

from __future__ import annotations

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # index of the feature to split on
        self.threshold = threshold  # split point: go left if x[feature] <= threshold
        self.left = left
        self.right = right
        self.value = value          # prediction at a leaf (None for internal nodes)

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


def _gini(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def _variance(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.var(y))


class DecisionTree:
    """CART decision tree.

    Parameters
    ----------
    task : {"classification", "regression"}
        Chooses the impurity (Gini vs variance) and the leaf prediction
        (majority vote vs mean).
    max_depth : int or None
        Maximum tree depth. ``None`` means grow until other stop criteria hit.
    min_samples_split : int
        Minimum samples required to consider splitting a node.
    """

    def __init__(self, task: str = "classification", max_depth=None, min_samples_split: int = 2):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ---- impurity / leaf helpers -------------------------------------------------

    def _impurity(self, y: np.ndarray) -> float:
        return _gini(y) if self.task == "classification" else _variance(y)

    def _leaf_value(self, y: np.ndarray):
        if self.task == "classification":
            vals, counts = np.unique(y, return_counts=True)
            return vals[np.argmax(counts)]
        return float(np.mean(y))

    # ---- split search ------------------------------------------------------------

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        parent_imp = self._impurity(y)
        best_gain = 0.0
        best = None  # (feature, threshold)

        for feature in range(n_features):
            values = X[:, feature]
            # Candidate thresholds: midpoints between consecutive unique values.
            uniq = np.unique(values)
            if uniq.size <= 1:
                continue
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0
            for thr in thresholds:
                left_mask = values <= thr
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                imp_left = self._impurity(y[left_mask])
                imp_right = self._impurity(y[~left_mask])
                child_imp = (n_left * imp_left + n_right * imp_right) / n_samples
                gain = parent_imp - child_imp
                if gain > best_gain:
                    best_gain = gain
                    best = (feature, float(thr))
        return best, best_gain

    # ---- recursion ---------------------------------------------------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = X.shape[0]
        # Stop: pure node, too few samples, or depth cap reached.
        stop = (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.unique(y).size == 1
        )
        if not stop:
            split, gain = self._best_split(X, y)
            if split is not None and gain > 0.0:
                feature, thr = split
                left_mask = X[:, feature] <= thr
                left = self._build(X[left_mask], y[left_mask], depth + 1)
                right = self._build(X[~left_mask], y[~left_mask], depth + 1)
                return _Node(feature=feature, threshold=thr, left=left, right=right)
        return _Node(value=self._leaf_value(y))

    def fit(self, X, y) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.root = self._build(X, y, depth=0)
        return self

    # ---- prediction --------------------------------------------------------------

    def _predict_row(self, row: np.ndarray):
        node = self.root
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.array([self._predict_row(row) for row in X])
