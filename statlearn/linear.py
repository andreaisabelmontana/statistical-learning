"""Linear models implemented from first principles with numpy.

- LinearRegression : ordinary least squares, solved either by the normal
  equation (closed form) or by batch gradient descent.
- RidgeRegression  : L2-penalised least squares, closed form.
- LassoRegression  : L1-penalised least squares, solved by coordinate descent.

All three share a ``fit(X, y) -> self`` / ``predict(X) -> ndarray`` interface and
expose ``coef_`` (slopes) and ``intercept_`` (bias) after fitting. The intercept
is never regularised.
"""

from __future__ import annotations

import numpy as np

from .metrics import r2_score


def _as_matrix(X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


class LinearRegression:
    """Ordinary least squares.

    Parameters
    ----------
    method : {"normal", "gd"}
        ``"normal"`` solves the normal equation (X^T X) w = X^T y in closed
        form. ``"gd"`` runs batch gradient descent on the mean-squared-error
        loss.
    lr : float
        Learning rate for gradient descent.
    n_iters : int
        Number of gradient-descent steps.
    """

    def __init__(self, method: str = "normal", lr: float = 0.01, n_iters: int = 10000):
        if method not in ("normal", "gd"):
            raise ValueError("method must be 'normal' or 'gd'")
        self.method = method
        self.lr = lr
        self.n_iters = n_iters
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y) -> "LinearRegression":
        X = _as_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        # Prepend a column of ones so the intercept is learned with the weights.
        Xb = np.hstack([np.ones((n_samples, 1)), X])

        if self.method == "normal":
            # Solve (Xb^T Xb) w = Xb^T y. lstsq is the numerically stable route
            # and degrades gracefully to the minimum-norm solution if singular.
            w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        else:
            w = np.zeros(n_features + 1)
            for _ in range(self.n_iters):
                grad = (2.0 / n_samples) * Xb.T @ (Xb @ w - y)
                w -= self.lr * grad

        self.intercept_ = float(w[0])
        self.coef_ = w[1:]
        return self

    def predict(self, X) -> np.ndarray:
        X = _as_matrix(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y) -> float:
        return r2_score(y, self.predict(X))


class RidgeRegression:
    """L2-penalised least squares (ridge), closed form.

    Minimises ||y - Xw||^2 + alpha * ||w||^2, where the intercept is excluded
    from the penalty. Solved as w = (X^T X + alpha I')^{-1} X^T y on centred
    data, which keeps the bias term unregularised.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y) -> "RidgeRegression":
        X = _as_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        # Centre X and y so the intercept decouples from the penalised weights.
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        yc = y - y_mean

        n_features = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_features)
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X) -> np.ndarray:
        X = _as_matrix(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y) -> float:
        return r2_score(y, self.predict(X))


class LassoRegression:
    """L1-penalised least squares (lasso) via coordinate descent.

    Minimises (1 / 2n) * ||y - Xw||^2 + alpha * ||w||_1. Each coordinate is
    updated in closed form using the soft-thresholding operator; the intercept
    is handled by centring and is never penalised. Features are assumed roughly
    comparable in scale (standardise beforehand for best results).
    """

    def __init__(self, alpha: float = 1.0, n_iters: int = 1000, tol: float = 1e-6):
        self.alpha = float(alpha)
        self.n_iters = n_iters
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _soft_threshold(rho: float, lam: float) -> float:
        if rho < -lam:
            return rho + lam
        if rho > lam:
            return rho - lam
        return 0.0

    def fit(self, X, y) -> "LassoRegression":
        X = _as_matrix(X)
        y = np.asarray(y, dtype=float).ravel()
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        yc = y - y_mean

        n_samples, n_features = Xc.shape
        w = np.zeros(n_features)
        # Precompute the per-feature normaliser sum(x_j^2).
        col_sq = np.sum(Xc ** 2, axis=0)

        for _ in range(self.n_iters):
            w_old = w.copy()
            for j in range(n_features):
                if col_sq[j] == 0.0:
                    w[j] = 0.0
                    continue
                # Partial residual excluding feature j's current contribution.
                residual = yc - Xc @ w + w[j] * Xc[:, j]
                rho = Xc[:, j] @ residual
                w[j] = self._soft_threshold(rho, self.alpha * n_samples) / col_sq[j]
            if np.max(np.abs(w - w_old)) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self

    def predict(self, X) -> np.ndarray:
        X = _as_matrix(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y) -> float:
        return r2_score(y, self.predict(X))
