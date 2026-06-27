"""Validate k-fold splits and bootstrap against scikit-learn / known properties."""

import numpy as np
import pytest

from sklearn.model_selection import KFold

from statlearn import (
    LinearRegression,
    kfold_split,
    cross_val_score,
    bootstrap_sample,
    bootstrap_score,
)


def test_kfold_fold_sizes_match_sklearn():
    n, k = 23, 5  # not evenly divisible -> uneven fold sizes
    ours = [len(test) for _, test in kfold_split(n, k, shuffle=False)]
    sk = [len(test) for _, test in KFold(n_splits=k, shuffle=False).split(np.arange(n))]
    assert ours == sk
    assert sum(ours) == n


def test_kfold_partitions_all_indices_once():
    n, k = 50, 7
    seen = []
    for train, test in kfold_split(n, k, shuffle=True, random_state=0):
        # Train and test are disjoint and together cover everything.
        assert set(train).isdisjoint(set(test))
        assert sorted(np.concatenate([train, test])) == list(range(n))
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(n))


def test_cross_val_score_runs_and_returns_per_fold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X @ np.array([1.0, -1.0, 2.0]) + rng.normal(scale=0.1, size=120)
    scores = cross_val_score(LinearRegression(), X, y, n_splits=5, scoring="r2", random_state=0)
    assert scores.shape == (5,)
    assert scores.mean() > 0.95


def test_bootstrap_oob_fraction_is_about_one_third():
    n = 1000
    fractions = []
    for s in range(50):
        _, oob = bootstrap_sample(n, random_state=s)
        fractions.append(oob.size / n)
    # Expected OOB fraction ~ 1/e ≈ 0.368.
    assert abs(np.mean(fractions) - 0.368) < 0.02


def test_bootstrap_in_bag_has_full_length():
    in_bag, oob = bootstrap_sample(100, random_state=7)
    assert in_bag.size == 100
    assert set(in_bag).isdisjoint(set(oob))


def test_bootstrap_score_runs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 2))
    y = X @ np.array([2.0, -3.0]) + rng.normal(scale=0.1, size=80)
    scores = bootstrap_score(LinearRegression(), X, y, n_rounds=30, scoring="r2", random_state=0)
    assert scores.size > 0
    assert scores.mean() > 0.9
