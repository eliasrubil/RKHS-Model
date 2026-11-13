"""Metrics for cost-sensitive credit scoring and subgroup analysis.

Conventions (shared by the whole project)
    * ``y = 1`` is a "bad" credit risk (the positive class), ``y = 0`` is "good".
    * A *false negative* is a bad applicant classified as good (the expensive
      error); a *false positive* is a good applicant classified as bad.
    * Scores are oriented so that higher means riskier.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class CostMatrix:
    """Misclassification costs of the Statlog German Credit documentation."""

    false_negative: float  # bad clasificado como good
    false_positive: float  # good clasificado como bad

    @classmethod
    def from_config(cls, config: Mapping) -> CostMatrix:
        cost_cfg = config["cost"]
        return cls(float(cost_cfg["false_negative"]), float(cost_cfg["false_positive"]))

    def class_weight(self) -> dict[int, float]:
        """Class weights that make the training loss mirror the cost matrix."""
        # Equivocarse con un ejemplo de la clase k cuesta w_k: Ec. (A.35) [eq:class-weights].
        return {1: self.false_negative, 0: self.false_positive}

    def total(self, y_true, y_pred) -> float:
        """Total cost ``c_FN * FN + c_FP * FP`` of a set of predictions."""
        _, fp, fn, _ = _confusion_counts(y_true, y_pred)
        return self.false_negative * fn + self.false_positive * fp


def _confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return (TN, FP, FN, TP) with the positive class equal to 1."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def classification_metrics(y_true, y_pred, scores, cost: CostMatrix) -> dict:
    """Metrics reported for every model.

    ``scores`` may be ``None`` (or constant) for classifiers without a ranking;
    the AUC is then reported as 0.5, the value of a random ranking.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    if scores is None or np.ptp(scores) == 0 or np.unique(y_true).size < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, scores))
    total_cost = cost.total(y_true, y_pred)
    return {
        "n": int(y_true.size),
        "auc": auc,
        "recall_bad": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_bad": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "cost": float(total_cost),
        "cost_per_sample": float(total_cost / y_true.size),
    }


def subgroup_metrics(y_true, y_pred, groups: pd.Series) -> pd.DataFrame:
    """Approval rate and error rates for every value of a sensitive attribute.

    Columns
    -------
    n, n_bad
        Group size and number of bad applicants in it (small counts make the
        rates unreliable; they are reported so the reader can judge).
    approval_rate
        Fraction predicted "good" (i.e. the credit would be granted).
    recall_bad
        Fraction of the bad applicants of the group that the model flags.
    false_positive_rate
        Fraction of the good applicants of the group that are wrongly rejected.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_values = np.asarray(groups)
    rows = []
    for value in sorted(pd.unique(group_values)):
        mask = group_values == value
        tn, fp, fn, tp = _confusion_counts(y_true[mask], y_pred[mask])
        n_bad, n_good = tp + fn, tn + fp
        rows.append(
            {
                "group": str(value),
                "n": int(mask.sum()),
                "n_bad": n_bad,
                "base_rate_bad": n_bad / mask.sum(),
                "approval_rate": float(np.mean(y_pred[mask] == 0)),
                "recall_bad": tp / n_bad if n_bad else float("nan"),
                "false_positive_rate": fp / n_good if n_good else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def approval_rate_gap(table: pd.DataFrame) -> float:
    """Largest difference in approval rate between groups (demographic parity gap)."""
    return float(table["approval_rate"].max() - table["approval_rate"].min())


def agreement(predictions_a, predictions_b) -> float:
    """Fraction of identical predictions between two classifiers."""
    return float(np.mean(np.asarray(predictions_a) == np.asarray(predictions_b)))
