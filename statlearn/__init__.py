"""statlearn — core statistical-learning methods implemented from scratch with numpy.

Public API:
    LinearRegression   — OLS via normal equation or gradient descent
    RidgeRegression    — L2-penalised least squares (closed form)
    LassoRegression    — L1-penalised least squares (coordinate descent)
    LogisticRegression — binary classifier via gradient descent
    DecisionTree       — CART for regression or classification
    kfold_split, cross_val_score, bootstrap_sample, bootstrap_score
    r2_score, mean_squared_error, accuracy_score
"""

from .linear import LinearRegression, RidgeRegression, LassoRegression
from .logistic import LogisticRegression
from .tree import DecisionTree
from .resampling import (
    kfold_split,
    cross_val_score,
    bootstrap_sample,
    bootstrap_score,
)
from .metrics import r2_score, mean_squared_error, accuracy_score

__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "LogisticRegression",
    "DecisionTree",
    "kfold_split",
    "cross_val_score",
    "bootstrap_sample",
    "bootstrap_score",
    "r2_score",
    "mean_squared_error",
    "accuracy_score",
]

__version__ = "0.1.0"
