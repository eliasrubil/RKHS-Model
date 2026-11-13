"""Cost-sensitive metrics and subgroup analysis on hand-computed examples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rkhs_credit.evaluation import (
    CostMatrix,
    agreement,
    approval_rate_gap,
    classification_metrics,
    subgroup_metrics,
)

COST = CostMatrix(false_negative=5.0, false_positive=1.0)

#                 TN TN TN FP FP FN TP TP
Y_TRUE = np.array([0, 0, 0, 0, 0, 1, 1, 1])
Y_PRED = np.array([0, 0, 0, 1, 1, 0, 1, 1])


def test_cost_matrix_total_and_class_weights():
    assert COST.total(Y_TRUE, Y_PRED) == 5.0 * 1 + 1.0 * 2
    assert COST.class_weight() == {1: 5.0, 0: 1.0}
    config = {"cost": {"false_negative": 5, "false_positive": 1}}
    assert CostMatrix.from_config(config) == COST


def test_classification_metrics_on_a_known_confusion_matrix():
    scores = np.array([-2.0, -1.5, -1.0, 0.5, 0.2, -0.1, 1.0, 2.0])
    metrics = classification_metrics(Y_TRUE, Y_PRED, scores, COST)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (3, 2, 1, 2)
    assert metrics["recall_bad"] == pytest.approx(2 / 3)
    assert metrics["precision_bad"] == pytest.approx(2 / 4)
    assert metrics["accuracy"] == pytest.approx(5 / 8)
    assert metrics["cost"] == 7.0
    assert metrics["cost_per_sample"] == pytest.approx(7 / 8)
    # 15 pares (bad, good); solo en 2 el "bad" tiene menor puntaje que el "good".
    assert metrics["auc"] == pytest.approx(13 / 15)


def test_accuracy_can_hide_the_most_expensive_classifier():
    """Approving everyone beats rejecting everyone in accuracy but costs far more."""
    y_true = np.array([0] * 70 + [1] * 30)
    approve_all = classification_metrics(y_true, np.zeros(100, dtype=int), None, COST)
    reject_all = classification_metrics(y_true, np.ones(100, dtype=int), None, COST)
    assert approve_all["accuracy"] == 0.7 > reject_all["accuracy"] == 0.3
    assert approve_all["cost"] == 150.0 > reject_all["cost"] == 70.0
    assert approve_all["auc"] == reject_all["auc"] == 0.5
    assert approve_all["recall_bad"] == 0.0 and approve_all["precision_bad"] == 0.0


def test_subgroup_metrics_and_approval_gap():
    groups = pd.Series(["a", "a", "a", "b", "b", "a", "b", "b"])
    table = subgroup_metrics(Y_TRUE, Y_PRED, groups)
    # Grupo a: y = [0, 0, 0, 1], pred = [0, 0, 0, 0]
    assert table.loc["a", "n"] == 4 and table.loc["a", "n_bad"] == 1
    assert table.loc["a", "approval_rate"] == 1.0
    assert table.loc["a", "recall_bad"] == 0.0
    assert table.loc["a", "false_positive_rate"] == 0.0
    # Grupo b: y = [0, 0, 1, 1], pred = [1, 1, 1, 1]
    assert table.loc["b", "approval_rate"] == 0.0
    assert table.loc["b", "recall_bad"] == 1.0
    assert table.loc["b", "false_positive_rate"] == 1.0
    assert table.loc["b", "base_rate_bad"] == 0.5
    assert approval_rate_gap(table) == 1.0


def test_subgroup_without_bad_applicants_reports_nan_recall():
    table = subgroup_metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), pd.Series(["x", "x", "y"]))
    assert np.isnan(table.loc["x", "recall_bad"])
    assert table.loc["x", "false_positive_rate"] == 0.5
    assert np.isnan(table.loc["y", "false_positive_rate"])


def test_agreement():
    assert agreement([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
