# statistical-learning

The core statistical-learning methods — linear and logistic regression, regularisation,
trees, and resampling — implemented from first principles in Python with **numpy only**.
No scikit-learn is used for any of the algorithms; it appears solely in the test suite and
demo to generate datasets and to cross-check the from-scratch code for correctness.

- Live page: https://andreaisabelmontana.github.io/statistical-learning/

## What's implemented

| Method | File | How it works |
| --- | --- | --- |
| Linear regression | `statlearn/linear.py` | OLS two ways: the **normal equation** (closed form, via `lstsq`) and **batch gradient descent** on the MSE loss. |
| Ridge regression | `statlearn/linear.py` | L2-penalised least squares, **closed form** on centred data so the intercept stays unpenalised: `w = (XᵀX + αI)⁻¹Xᵀy`. |
| Lasso regression | `statlearn/linear.py` | L1-penalised least squares via **coordinate descent** with the soft-thresholding operator — drives weak coefficients to exactly zero. |
| Logistic regression | `statlearn/logistic.py` | Binary classifier trained by **gradient descent** on cross-entropy, with a numerically stable sigmoid, optional L2, and `predict_proba`. |
| Decision tree (CART) | `statlearn/tree.py` | Greedy recursive binary splits. Classification uses **Gini impurity**; regression uses **variance reduction**. Exhaustive threshold scan. |
| k-fold cross-validation | `statlearn/resampling.py` | Index-based folds (sizes differ by ≤1, matching sklearn's `KFold`) plus a `cross_val_score` driver. |
| Bootstrap | `statlearn/resampling.py` | Resample with replacement; held-out **out-of-bag** rows give a generalisation estimate (`bootstrap_score`). |

Every estimator shares the same `fit(X, y)` / `predict(X)` interface and exposes
`coef_` / `intercept_` (linear models) after fitting.

## Install

```bash
pip install -r requirements.txt
```

`numpy` is the only runtime dependency. `scikit-learn` and `pytest` are dev-only.

## Run the tests

The suite cross-checks each from-scratch implementation against its scikit-learn
counterpart within a numerical tolerance (OLS/Ridge/Lasso coefficients, logistic
accuracy and coefficients, CART accuracy/R², and k-fold split sizes vs `KFold`).

```bash
python -m pytest -q
```

```
20 passed in 7.33s
```

## Run the demo

```bash
python demo.py
```

The demo trains every method on two standard datasets (`diabetes` for regression,
`breast_cancer` for classification), using scikit-learn only to load the data.
Real output from a run on this machine:

```
REGRESSION  -  diabetes dataset (442 samples, 10 features)
OLS (normal eq.)   test R^2 = 0.3728
OLS (grad descent) test R^2 = 0.3728
Ridge (alpha=10)   test R^2 = 0.3696
Lasso (alpha=0.5)  test R^2 = 0.3733   (9/10 features kept)
Decision tree d=3  test R^2 = 0.2043
OLS 5-fold CV R^2  = 0.4875 +/- 0.0751

CLASSIFICATION  -  breast cancer dataset (569 samples, 30 features)
Logistic regression  test accuracy = 0.9580
Decision tree d=4    test accuracy = 0.9231
Logistic 5-fold CV   accuracy = 0.9736 +/- 0.0124
Tree OOB bootstrap   accuracy = 0.9207 +/- 0.0177
```

Two things worth noting from these numbers: the normal-equation and gradient-descent
solvers land on an identical test R² (0.3728), which is the convergence check working as
intended; and lasso at `alpha=0.5` zeroes out one of the ten diabetes features while
holding the same accuracy — regularisation doing feature selection.

## Example

```python
import numpy as np
from statlearn import LinearRegression, LogisticRegression, cross_val_score

X = np.random.randn(200, 3)
y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * np.random.randn(200)

model = LinearRegression(method="normal").fit(X, y)
print(model.coef_, model.intercept_)
print(model.score(X, y))                 # R^2 on the training data

scores = cross_val_score(LinearRegression(), X, y, n_splits=5, scoring="r2")
print(scores.mean())
```

## Layout

```
statlearn/
  linear.py      OLS, ridge, lasso
  logistic.py    logistic regression
  tree.py        CART decision tree
  resampling.py  k-fold CV + bootstrap
  metrics.py     r2_score, accuracy_score, mean_squared_error
tests/           pytest suite cross-checking against scikit-learn
demo.py          end-to-end run on real datasets
```

## License

MIT — see [LICENSE](LICENSE).
