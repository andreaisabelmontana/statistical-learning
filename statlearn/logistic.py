"""Binary logistic regression via batch gradient descent (numpy, from scratch)."""

from __future__ import annotations

import numpy as np

from .metrics import accuracy_score


def _as_matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable logistic function (no overflow for large |z|).
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """Logistic regression for binary labels {0, 1}.

    Minimises the average binary cross-entropy with an optional L2 penalty
    (``C`` is the inverse regularisation strength, matching the sklearn
    convention; larger C means weaker regularisation). Optimised by batch
    gradient descent.

    Parameters
    ----------
    lr : float
        Learning rate.
    n_iters : int
        Number of gradient-descent steps.
    C : float
        Inverse L2 regularisation strength. ``np.inf`` disables the penalty.
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 5000, C: float = 1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y) -> "LogisticRegression":
        X = _as_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        Xb = np.hstack([np.ones((n_samples, 1)), X])
        w = np.zeros(n_features + 1)

        # L2 strength per the sklearn parametrisation: penalty weight = 1 / C,
        # applied to the weights but not the intercept.
        lam = 0.0 if np.isinf(self.C) else 1.0 / self.C

        for _ in range(self.n_iters):
            p = _sigmoid(Xb @ w)
            grad = Xb.T @ (p - y) / n_samples
            if lam:
                reg = lam * w / n_samples
                reg[0] = 0.0  # never regularise the intercept
                grad = grad + reg
            w -= self.lr * grad

        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Probability of the positive class for each row."""
        X = _as_matrix(X)
        return _sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y) -> float:
        return accuracy_score(y, self.predict(X))
