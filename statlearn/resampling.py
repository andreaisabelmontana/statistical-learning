"""Resampling utilities: k-fold cross-validation and the bootstrap (from scratch)."""

from __future__ import annotations

import numpy as np

from .metrics import r2_score, accuracy_score


def kfold_split(n_samples: int, n_splits: int = 5, shuffle: bool = True, random_state=None):
    """Yield (train_index, test_index) pairs for k-fold cross-validation.

    Fold sizes differ by at most one: the first ``n_samples % n_splits`` folds
    get one extra sample, matching scikit-learn's KFold behaviour.
    """
    if n_splits < 2 or n_splits > n_samples:
        raise ValueError("n_splits must be between 2 and n_samples")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current : current + size])
        current += size

    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, test_idx


def cross_val_score(model, X, y, n_splits=5, scoring="r2", shuffle=True, random_state=None):
    """Clone-free k-fold CV. The same model object is re-fit on each fold.

    ``scoring`` is "r2" or "accuracy". Returns an array of per-fold scores.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if scoring == "r2":
        score_fn = r2_score
    elif scoring == "accuracy":
        score_fn = accuracy_score
    else:
        raise ValueError("scoring must be 'r2' or 'accuracy'")

    scores = []
    for train_idx, test_idx in kfold_split(len(y), n_splits, shuffle, random_state):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(score_fn(y[test_idx], preds))
    return np.array(scores)


def bootstrap_sample(n_samples: int, random_state=None):
    """Return (in_bag_index, out_of_bag_index) for one bootstrap resample.

    ``in_bag`` is sampled with replacement (length n_samples, may repeat);
    ``out_of_bag`` are the indices never drawn (~36.8% of the data on average).
    """
    rng = np.random.default_rng(random_state)
    in_bag = rng.integers(0, n_samples, size=n_samples)
    mask = np.ones(n_samples, dtype=bool)
    mask[in_bag] = False
    out_of_bag = np.arange(n_samples)[mask]
    return in_bag, out_of_bag


def bootstrap_score(model, X, y, n_rounds=200, scoring="r2", random_state=None):
    """Out-of-bag bootstrap estimate of generalisation error.

    Fits the model on each bootstrap resample and scores on the held-out OOB
    rows. Returns an array of per-round scores (rounds with empty OOB skipped).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if scoring == "r2":
        score_fn = r2_score
    elif scoring == "accuracy":
        score_fn = accuracy_score
    else:
        raise ValueError("scoring must be 'r2' or 'accuracy'")

    rng = np.random.default_rng(random_state)
    n = len(y)
    scores = []
    for _ in range(n_rounds):
        seed = int(rng.integers(0, 2**31 - 1))
        in_bag, oob = bootstrap_sample(n, random_state=seed)
        if oob.size == 0:
            continue
        model.fit(X[in_bag], y[in_bag])
        scores.append(score_fn(y[oob], model.predict(X[oob])))
    return np.array(scores)
