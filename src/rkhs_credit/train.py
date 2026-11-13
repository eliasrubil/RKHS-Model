"""Train and evaluate every model of the project.

Usage::

    python -m rkhs_credit.train --config configs/default.yaml

Steps
-----
1. Load German Credit (cached) and make the stratified train/test split.
2. Choose hyper-parameters by stratified k-fold cross-validation on train.
3. Fit the kernel SVM, the baselines and the configured variants on train.
4. Evaluate everything once on the held-out test set.
5. Check the hand-written SVM against ``sklearn.svm.SVC``.
6. Report spectral diagnostics (condition number) of the Gram matrix.
7. Sensitive-attribute analysis: model with / without them, metrics per group.
8. Write ``metrics.json``, CSV tables and LaTeX snippets to ``paths.results_dir``.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from rkhs_credit.config import load_config, with_overrides
from rkhs_credit.data import data_source, derive_sensitive_groups, load_and_split
from rkhs_credit.evaluation import (
    CostMatrix,
    agreement,
    approval_rate_gap,
    classification_metrics,
    subgroup_metrics,
)
from rkhs_credit.numerics import center_gram, spectral_summary
from rkhs_credit.pipeline import (
    ConstantClassifier,
    MajorityClassifier,
    SklearnSVCReference,
    build_logistic_baseline,
    build_pipeline,
)
from rkhs_credit.reporting import MODEL_LABELS, write_outputs
from rkhs_credit.tuning import ModelBuilder, cross_validate, grid_search, make_folds, select_best

logger = logging.getLogger("rkhs_credit.train")


def tune(
    config: Mapping,
    grid: Mapping[str, Sequence] | None,
    X: pd.DataFrame,
    y: np.ndarray,
    builder: ModelBuilder = build_pipeline,
) -> tuple[dict, dict, pd.DataFrame]:
    """Choose hyper-parameters by cross-validation.

    Returns ``(tuned_config, cv_summary, cv_table)``.  With ``cv.enabled: false``
    (or an empty grid) the configuration is returned unchanged and only its
    own cross-validation score is computed.
    """
    cv_cfg = config["cv"]
    if cv_cfg.get("enabled", True) and grid:
        table = grid_search(config, grid, X, y, builder)
        best_params = select_best(table, grid, cv_cfg["scoring"])
        tuned = with_overrides(config, best_params)
        mask = np.ones(len(table), dtype=bool)
        for key, value in best_params.items():
            mask &= (table[key] == value).to_numpy()
        best_row = table[mask].iloc[0]
        scores = {key: float(best_row[key]) for key in ("mean_cost", "std_cost", "mean_auc")}
        scores["n_not_converged"] = int(table["n_not_converged"].sum())
    else:
        folds = make_folds(y, cv_cfg["n_splits"], config["seed"])
        scores = cross_validate(config, X, y, folds, builder)
        table = pd.DataFrame([scores])
        best_params = {}
        tuned = with_overrides(config, None)
    summary = {"best_params": best_params, **scores}
    return tuned, summary, table


def evaluate(model, X: pd.DataFrame, y: np.ndarray, cost: CostMatrix) -> tuple[dict, dict]:
    """Test metrics plus the raw predictions and scores of a fitted model."""
    predictions = model.predict(X)
    scores = model.decision_function(X)
    return classification_metrics(y, predictions, scores, cost), {
        "prediction": predictions,
        "score": scores,
    }


def run(config: Mapping) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Run the whole experiment; returns ``(results, tables)``."""
    start = time.perf_counter()
    cost = CostMatrix.from_config(config)
    X_train, X_test, y_train, y_test = load_and_split(config)
    logger.info(
        "Data: %d train / %d test, bad rate %.3f", len(X_train), len(X_test), y_train.mean()
    )

    labels = dict(MODEL_LABELS)
    tables: dict[str, pd.DataFrame] = {}
    cv_summary: dict[str, dict] = {}
    test_metrics: dict[str, dict] = {}
    outputs: dict[str, dict] = {}

    # ---- 1. baselines triviales -----------------------------------------
    for name, model in (("majority", MajorityClassifier()), ("always_bad", ConstantClassifier(1))):
        test_metrics[name], outputs[name] = evaluate(
            model.fit(X_train, y_train), X_test, y_test, cost
        )

    # ---- 2. regresion logistica -----------------------------------------
    logreg_grid = config["baselines"]["logistic_regression"].get("grid")
    logreg_config, cv_summary["logreg"], tables["cv_logreg"] = tune(
        config, logreg_grid, X_train, y_train, build_logistic_baseline
    )
    logreg = build_logistic_baseline(logreg_config).fit(X_train, y_train)
    test_metrics["logreg"], outputs["logreg"] = evaluate(logreg, X_test, y_test, cost)

    # ---- 3. SVM propia ---------------------------------------------------
    logger.info("Cross-validating the kernel SVM ...")
    svm_config, cv_summary["svm"], tables["cv_svm"] = tune(
        config, config["cv"].get("grid"), X_train, y_train
    )
    fit_start = time.perf_counter()
    pipeline = build_pipeline(svm_config).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    test_metrics["svm"], outputs["svm"] = evaluate(pipeline, X_test, y_test, cost)
    svm = pipeline.svm

    # ---- 4. variantes (p. ej. Kernel PCA + SVM lineal) --------------------
    for name, variant in (config.get("variants") or {}).items():
        logger.info("Cross-validating variant %r ...", name)
        labels[name] = variant.get("label", name)
        variant_config = with_overrides(svm_config, variant.get("overrides"))
        variant_config, cv_summary[name], tables[f"cv_{name}"] = tune(
            variant_config, variant.get("grid"), X_train, y_train
        )
        variant_model = build_pipeline(variant_config).fit(X_train, y_train)
        test_metrics[name], outputs[name] = evaluate(variant_model, X_test, y_test, cost)

    # ---- 5. verificacion contra scikit-learn ------------------------------
    reference = SklearnSVCReference(svm, exclude=pipeline.exclude).fit(X_train, y_train)
    reference_scores = reference.decision_function(X_test)
    sklearn_check = {
        "prediction_agreement": agreement(outputs["svm"]["prediction"], reference.predict(X_test)),
        "decision_correlation": float(np.corrcoef(outputs["svm"]["score"], reference_scores)[0, 1]),
        "max_abs_decision_difference": float(
            np.max(np.abs(outputs["svm"]["score"] - reference_scores))
        ),
        "n_support_sklearn": int(reference.model_.support_.size),
        "test_metrics_sklearn": classification_metrics(
            y_test, reference.predict(X_test), reference_scores, cost
        ),
    }

    # ---- 6. diagnostico espectral de la matriz de Gram --------------------
    features = pipeline.preprocessor_.transform(X_train)
    gram = svm.kernel.gram(features)
    gram_summary = spectral_summary(gram, jitter=svm.jitter)
    centered_summary = spectral_summary(center_gram(gram))

    results: dict = {
        "data": {
            "source": data_source(config),
            "n_samples": len(X_train) + len(X_test),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_test_bad": int(y_test.sum()),
            "n_test_good": int((1 - y_test).sum()),
            "n_attributes": X_train.shape[1],
            "n_encoded_features": int(features.shape[1]),
            "bad_rate_train": float(y_train.mean()),
            "bad_rate_test": float(y_test.mean()),
        },
        "seed": config["seed"],
        "labels": labels,
        "cost_matrix": {
            "false_negative": cost.false_negative,
            "false_positive": cost.false_positive,
        },
        "cv": cv_summary,
        "test": test_metrics,
        "svm_fit": {
            "kernel": repr(svm.kernel),
            "C": svm.C,
            "class_weight": {str(k): v for k, v in (svm.class_weight or {}).items()},
            "tol": svm.tol,
            "n_support": int(svm.support_.size),
            "n_free_support": svm.n_free_support_,
            "intercept": svm.intercept_,
            "n_iter": svm.n_iter_,
            "converged": bool(svm.converged_),
            "kkt_gap": svm.kkt_gap_,
            "dual_objective": svm.dual_objective_,
            "fit_seconds": fit_seconds,
        },
        "sklearn_check": sklearn_check,
        "gram": gram_summary.to_dict(),
        "centered_gram": centered_summary.to_dict(),
    }

    # ---- 7. atributos sensibles -------------------------------------------
    predictions = pd.DataFrame({"y_true": y_test}, index=X_test.index)
    sensitive_cfg = config.get("sensitive_analysis") or {}
    if sensitive_cfg.get("enabled", False):
        results["sensitive"], sensitive_predictions = sensitive_analysis(
            config, svm_config, cv_summary["svm"], X_train, y_train, X_test, y_test
        )
        predictions = predictions.join(sensitive_predictions)
    for name, output in outputs.items():
        predictions[f"{name}_prediction"] = output["prediction"]
        predictions[f"{name}_score"] = output["score"]
    predictions.index.name = "row"
    tables["test_predictions"] = predictions

    results["runtime_seconds"] = time.perf_counter() - start
    return results, tables


def sensitive_analysis(
    config: Mapping,
    svm_config: Mapping,
    svm_cv_summary: Mapping,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Compare the SVM with and without the sensitive attributes.

    The subgroup labels are always computed from the raw test attributes, also
    for the model that is not allowed to *use* them.
    """
    sensitive_cfg = config["sensitive_analysis"]
    cost = CostMatrix.from_config(config)
    groups = derive_sensitive_groups(X_test, sensitive_cfg["age_threshold"])
    main_excludes = bool(config["data"].get("exclude_sensitive", False))

    analysis: dict = {"age_threshold": sensitive_cfg["age_threshold"]}
    predictions = groups.copy()
    for key, exclude in (("with", False), ("without", True)):
        variant_config = with_overrides(svm_config, {"data.exclude_sensitive": exclude})
        if exclude == main_excludes:
            # Es el modelo principal: sus hiperparametros ya fueron elegidos por CV.
            best_params = dict(svm_cv_summary["best_params"])
        elif sensitive_cfg.get("retune", False):
            logger.info("Cross-validating the SVM %s sensitive attributes ...", key)
            base = with_overrides(config, {"data.exclude_sensitive": exclude})
            variant_config, summary, _ = tune(base, config["cv"].get("grid"), X_train, y_train)
            best_params = summary["best_params"]
        else:
            best_params = dict(svm_cv_summary["best_params"])
        model = build_pipeline(variant_config).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        group_tables = {
            attribute: subgroup_metrics(y_test, y_pred, groups[attribute])
            for attribute in groups.columns
        }
        analysis[key] = {
            "best_params": best_params,
            "test": classification_metrics(y_test, y_pred, model.decision_function(X_test), cost),
            "groups": {
                attribute: table.to_dict(orient="index")
                for attribute, table in group_tables.items()
            },
            "approval_gap": {
                attribute: approval_rate_gap(table) for attribute, table in group_tables.items()
            },
        }
        predictions[f"svm_{key}_sensitive_prediction"] = y_pred
    return analysis, predictions


def print_summary(results: Mapping) -> None:
    """Print the test-set comparison as a plain-text table."""
    columns = ["auc", "recall_bad", "precision_bad", "accuracy", "fn", "fp", "cost"]
    table = pd.DataFrame(results["test"]).T[columns]
    print("\nTest set (cost = 5*FN + 1*FP unless reconfigured):")
    print(table.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nBest hyper-parameters:")
    for name, summary in results["cv"].items():
        print(f"  {name:10s} {summary['best_params']}  CV cost = {summary['mean_cost']:.3f}")
    fit, check, gram = results["svm_fit"], results["sklearn_check"], results["gram"]
    print(
        f"\nSVM: {fit['n_support']} support vectors ({fit['n_free_support']} free), "
        f"{fit['n_iter']} SMO iterations, KKT gap {fit['kkt_gap']:.2e}, "
        f"converged={fit['converged']}"
    )
    print(
        f"Agreement with sklearn SVC: {100 * check['prediction_agreement']:.1f} % of predictions, "
        f"decision-function correlation {check['decision_correlation']:.6f}"
    )
    print(
        f"Gram matrix: cond = {gram['condition_number']:.3e}, "
        f"with jitter {gram['jitter']:g}: {gram['condition_number_with_jitter']:.3e}, "
        f"numerical rank {gram['numerical_rank']}/{gram['n']}"
    )
    print(f"Total runtime: {results['runtime_seconds']:.1f} s")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML file.")
    parser.add_argument(
        "--quiet", action="store_true", help="Do not log cross-validation progress."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    warnings.simplefilter("always", RuntimeWarning)
    config = load_config(args.config)
    results, tables = run(config)
    results_dir = Path(config["paths"]["results_dir"])
    write_outputs(results, tables, results_dir)
    print_summary(results)
    print(f"Results written to {results_dir}/")


if __name__ == "__main__":
    main()
