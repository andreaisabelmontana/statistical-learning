"""Train every method in statlearn on a small dataset and print real metrics.

Run:  python demo.py

Uses scikit-learn ONLY to load/generate datasets (not for the algorithms).
"""

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer

from statlearn import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LogisticRegression,
    DecisionTree,
    cross_val_score,
    bootstrap_score,
)
from statlearn.metrics import r2_score, accuracy_score


def standardize(train, test):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def split(X, y, frac=0.75, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(frac * len(y))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def regression_demo():
    banner("REGRESSION  —  diabetes dataset (442 samples, 10 features)")
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = split(X, y, seed=0)
    Xtr, Xte = standardize(Xtr, Xte)

    ols = LinearRegression(method="normal").fit(Xtr, ytr)
    print(f"OLS (normal eq.)   test R^2 = {r2_score(yte, ols.predict(Xte)):.4f}")

    gd = LinearRegression(method="gd", lr=0.05, n_iters=20000).fit(Xtr, ytr)
    print(f"OLS (grad descent) test R^2 = {r2_score(yte, gd.predict(Xte)):.4f}")

    ridge = RidgeRegression(alpha=10.0).fit(Xtr, ytr)
    print(f"Ridge (alpha=10)   test R^2 = {r2_score(yte, ridge.predict(Xte)):.4f}")

    lasso = LassoRegression(alpha=0.5, n_iters=5000).fit(Xtr, ytr)
    nz = int(np.sum(np.abs(lasso.coef_) > 1e-6))
    print(f"Lasso (alpha=0.5)  test R^2 = {r2_score(yte, lasso.predict(Xte)):.4f}"
          f"   ({nz}/{X.shape[1]} features kept)")

    tree = DecisionTree(task="regression", max_depth=3).fit(Xtr, ytr)
    print(f"Decision tree d=3  test R^2 = {r2_score(yte, tree.predict(Xte)):.4f}")

    cv = cross_val_score(LinearRegression(), X, y, n_splits=5, scoring="r2", random_state=0)
    print(f"OLS 5-fold CV R^2  = {cv.mean():.4f} +/- {cv.std():.4f}")


def classification_demo():
    banner("CLASSIFICATION  —  breast cancer dataset (569 samples, 30 features)")
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = split(X, y, seed=0)
    Xtr, Xte = standardize(Xtr, Xte)

    logit = LogisticRegression(lr=0.5, n_iters=20000, C=1.0).fit(Xtr, ytr)
    print(f"Logistic regression  test accuracy = {accuracy_score(yte, logit.predict(Xte)):.4f}")

    tree = DecisionTree(task="classification", max_depth=4).fit(Xtr, ytr)
    print(f"Decision tree d=4    test accuracy = {accuracy_score(yte, tree.predict(Xte)):.4f}")

    cv = cross_val_score(
        LogisticRegression(lr=0.5, n_iters=8000),
        np.vstack([Xtr, Xte]),
        np.concatenate([ytr, yte]),
        n_splits=5,
        scoring="accuracy",
        random_state=0,
    )
    print(f"Logistic 5-fold CV   accuracy = {cv.mean():.4f} +/- {cv.std():.4f}")

    boot = bootstrap_score(
        DecisionTree(task="classification", max_depth=4),
        Xtr, ytr, n_rounds=100, scoring="accuracy", random_state=0,
    )
    print(f"Tree OOB bootstrap   accuracy = {boot.mean():.4f} +/- {boot.std():.4f}")


if __name__ == "__main__":
    regression_demo()
    classification_demo()
    print()
