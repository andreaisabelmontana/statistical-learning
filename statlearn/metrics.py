"""Scoring helpers used across the library and demo."""

from __future__ import annotations

import numpy as np


def mean_squared_error(y_true, y_pred) -> float:
    """Mean of squared residuals."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred) -> float:
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def accuracy_score(y_true, y_pred) -> float:
    """Fraction of labels predicted correctly."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))
