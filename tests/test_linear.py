"""Cross-check the from-scratch linear models against scikit-learn."""

import numpy as np
import pytest

from sklearn.linear_model import (
    LinearRegression as SkLinear,
    Ridge as SkRidge,
    Lasso as SkLasso,
)

from statlearn import LinearRegression, RidgeRegression, LassoRegression


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    n, d = 200, 4
    X = rng.normal(size=(n, d))
    true_w = np.array([3.0, -2.0, 0.0, 1.5])
    y = X @ true_w + 0.5 + rng.normal(scale=0.1, size=n)
    return X, y


def test_ols_normal_matches_sklearn(reg_data):
    X, y = reg_data
    ours = LinearRegression(method="normal").fit(X, y)
    sk = SkLinear().fit(X, y)
    assert np.allclose(ours.coef_, sk.coef_, atol=1e-8)
    assert np.isclose(ours.intercept_, sk.intercept_, atol=1e-8)


def test_ols_gradient_descent_converges_to_normal(reg_data):
    X, y = reg_data
    gd = LinearRegression(method="gd", lr=0.1, n_iters=20000).fit(X, y)
    sk = SkLinear().fit(X, y)
    # Gradient descent should land very close to the closed-form solution.
    assert np.allclose(gd.coef_, sk.coef_, atol=1e-3)
    assert np.isclose(gd.intercept_, sk.intercept_, atol=1e-3)


def test_ridge_matches_sklearn(reg_data):
    X, y = reg_data
    alpha = 5.0
    ours = RidgeRegression(alpha=alpha).fit(X, y)
    sk = SkRidge(alpha=alpha, fit_intercept=True).fit(X, y)
    assert np.allclose(ours.coef_, sk.coef_, atol=1e-6)
    assert np.isclose(ours.intercept_, sk.intercept_, atol=1e-6)


def test_lasso_matches_sklearn(reg_data):
    X, y = reg_data
    alpha = 0.1
    ours = LassoRegression(alpha=alpha, n_iters=5000, tol=1e-9).fit(X, y)
    sk = SkLasso(alpha=alpha, fit_intercept=True, max_iter=50000, tol=1e-9).fit(X, y)
    assert np.allclose(ours.coef_, sk.coef_, atol=1e-3)
    assert np.isclose(ours.intercept_, sk.intercept_, atol=1e-3)


def test_lasso_zeros_irrelevant_feature(reg_data):
    X, y = reg_data
    # Feature index 2 has true weight 0; lasso should shrink it hardest.
    ours = LassoRegression(alpha=0.3, n_iters=5000).fit(X, y)
    assert abs(ours.coef_[2]) < abs(ours.coef_[0])
    assert abs(ours.coef_[2]) < 0.1


def test_predict_shape_and_r2(reg_data):
    X, y = reg_data
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert model.score(X, y) > 0.99
