"""Hyper-parameter search by stratified k-fold cross-validation.

The search is a plain, explicit double loop (parameter combinations x folds).
For every fold the model is rebuilt from the configuration and fitted on the
training part of the fold only, so the preprocessing statistics never see the
validation part.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from rkhs_credit.config import with_overrides
from rkhs_credit.evaluation import CostMatrix, classification_metrics
from rkhs_credit.pipeline import build_pipeline

logger = logging.getLogger(__name__)

ModelBuilder = Callable[[Mapping], object]


def make_folds(y, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified folds as a list of (train_indices, validation_indices)."""
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def expand_grid(grid: Mapping[str, Sequence]) -> list[dict]:
    """Cartesian product of a ``{dotted_key: [values]}`` grid."""
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def cross_validate(
    config: Mapping,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    builder: ModelBuilder = build_pipeline,
) -> dict:
    """Fit ``builder(config)`` on every fold and average the validation metrics."""
    cost = CostMatrix.from_config(config)
    fold_costs, fold_aucs, n_not_converged = [], [], 0
    for train_idx, valid_idx in folds:
        model = builder(config)
        with warnings.catch_warnings():
            # La falta de convergencia se cuenta abajo en vez de emitir un aviso por fold.
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(X.iloc[train_idx], y[train_idx])
        if not getattr(getattr(model, "svm", None), "converged_", True):
            n_not_converged += 1
        X_valid = X.iloc[valid_idx]
        metrics = classification_metrics(
            y[valid_idx], model.predict(X_valid), model.decision_function(X_valid), cost
        )
        fold_costs.append(metrics["cost_per_sample"])
        fold_aucs.append(metrics["auc"])
    return {
        "mean_cost": float(np.mean(fold_costs)),
        "std_cost": float(np.std(fold_costs)),
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "n_not_converged": n_not_converged,
    }


def grid_search(
    config: Mapping,
    grid: Mapping[str, Sequence],
    X: pd.DataFrame,
    y: np.ndarray,
    builder: ModelBuilder = build_pipeline,
) -> pd.DataFrame:
    """Cross-validate every combination of ``grid``.

    Returns one row per combination with the parameter values (columns named
    by their dotted keys) and the columns produced by :func:`cross_validate`.
    ``mean_cost`` is the average misclassification cost *per applicant*.
    """
    cv_cfg = config["cv"]
    folds = make_folds(y, cv_cfg["n_splits"], config["seed"])
    rows = []
    combinations = expand_grid(grid)
    for number, params in enumerate(combinations, start=1):
        scores = cross_validate(with_overrides(config, params), X, y, folds, builder)
        rows.append({**params, **scores})
        logger.info(
            "CV %3d/%d  %s  cost=%.4f  auc=%.4f",
            number,
            len(combinations),
            params,
            scores["mean_cost"],
            scores["mean_auc"],
        )
    return pd.DataFrame(rows)


def select_best(results: pd.DataFrame, grid: Mapping[str, Sequence], scoring: str) -> dict:
    """Pick the best row of a :func:`grid_search` table.

    ``scoring="cost"`` minimises the mean cost (ties: higher AUC);
    ``scoring="auc"`` maximises the mean AUC (ties: lower cost).
    Remaining ties are resolved by grid order.
    """
    if scoring == "cost":
        ordered = results.sort_values(
            ["mean_cost", "mean_auc"], ascending=[True, False], kind="stable"
        )
    elif scoring == "auc":
        ordered = results.sort_values(
            ["mean_auc", "mean_cost"], ascending=[False, True], kind="stable"
        )
    else:
        raise ValueError(f"Unknown scoring {scoring!r}; use 'cost' or 'auc'.")
    best = ordered.iloc[0]
    return {key: _to_python(best[key]) for key in grid}


def _to_python(value):
    """Convert numpy scalars to built-in types (YAML/JSON friendly)."""
    return value.item() if isinstance(value, np.generic) else value
