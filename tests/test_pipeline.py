"""Configuration, pipeline assembly, cross-validation and the two CLI scripts.

The end-to-end tests run the real ``train`` and ``make_figures`` code on a
synthetic table, with the data loader replaced by a stub (no network).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rkhs_credit import make_figures, train
from rkhs_credit.config import get_nested, load_config, set_nested, with_overrides
from rkhs_credit.data import stratified_split
from rkhs_credit.evaluation import CostMatrix
from rkhs_credit.kernels import LinearKernel, RBFKernel
from rkhs_credit.pipeline import (
    ConstantClassifier,
    MajorityClassifier,
    build_logistic_baseline,
    build_pipeline,
    resolve_class_weight,
)
from rkhs_credit.reporting import write_outputs
from rkhs_credit.tuning import expand_grid, grid_search, make_folds, select_best

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


# ------------------------------------------------------------------ config
def test_default_config_loads_and_has_no_missing_sections():
    config = load_config(DEFAULT_CONFIG)
    assert config["seed"] == 42
    assert config["data"]["test_size"] == 0.2
    assert config["cv"]["n_splits"] == 5
    assert config["cost"] == {"false_negative": 5.0, "false_positive": 1.0}
    assert set(config["data"]["sensitive_features"]) == {"personal_status", "age", "foreign_worker"}


def test_load_config_rejects_incomplete_files(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\n")
    with pytest.raises(ValueError, match="missing the required section"):
        load_config(path)


def test_nested_access_and_overrides_do_not_mutate_the_original():
    config = {"svm": {"C": 1.0, "kernel": {"name": "rbf", "gamma": 0.1}}}
    assert get_nested(config, "svm.kernel.gamma") == 0.1
    with pytest.raises(KeyError):
        get_nested(config, "svm.kernel.degree")
    changed = with_overrides(config, {"svm.kernel.gamma": 0.5, "svm.new.entry": 3})
    assert changed["svm"]["kernel"]["gamma"] == 0.5
    assert changed["svm"]["new"]["entry"] == 3
    assert config["svm"]["kernel"]["gamma"] == 0.1 and "new" not in config["svm"]
    set_nested(config, "svm.C", 2.0)
    assert config["svm"]["C"] == 2.0


# ---------------------------------------------------------------- pipeline
def test_build_pipeline_reads_every_hyperparameter_from_the_config(small_config):
    config = with_overrides(
        small_config,
        {"svm.C": 2.5, "svm.kernel": {"name": "rbf", "gamma": 0.07}, "svm.tol": 1e-4},
    )
    pipeline = build_pipeline(config)
    assert pipeline.kpca is None
    assert pipeline.svm.C == 2.5 and pipeline.svm.tol == 1e-4
    assert isinstance(pipeline.svm.kernel, RBFKernel) and pipeline.svm.kernel.gamma == 0.07
    assert pipeline.svm.class_weight == {1: 5.0, 0: 1.0}  # tomado de la matriz de costos
    assert pipeline.exclude == ()

    with_kpca = build_pipeline(
        with_overrides(
            config,
            {"kpca.enabled": True, "kpca.n_components": 4, "kpca.kernel": {"name": "linear"}},
        )  # fmt: skip
    )
    assert with_kpca.kpca.n_components == 4
    assert isinstance(with_kpca.kpca.kernel, LinearKernel)


def test_resolve_class_weight():
    cost = CostMatrix(5.0, 1.0)
    assert resolve_class_weight("cost_matrix", cost) == {1: 5.0, 0: 1.0}
    assert resolve_class_weight("balanced", cost) == "balanced"
    assert resolve_class_weight(None, cost) is None
    assert resolve_class_weight("none", cost) is None
    assert resolve_class_weight({"0": 1, "1": 3}, cost) == {0: 1.0, 1: 3.0}


def test_pipeline_fit_predict_and_beats_the_trivial_cost(small_config, credit_like_data):
    X, y = credit_like_data
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.25, seed=1)
    pipeline = build_pipeline(with_overrides(small_config, {"svm.kernel.gamma": 0.05}))
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    scores = pipeline.decision_function(X_test)
    assert predictions.shape == scores.shape == (len(X_test),)
    assert set(np.unique(predictions)) <= {0, 1}
    assert np.array_equal(predictions == 1, scores > 0)

    cost = CostMatrix.from_config(small_config)
    trivial = min(
        cost.total(y_test, np.zeros_like(y_test)), cost.total(y_test, np.ones_like(y_test))
    )
    assert cost.total(y_test, predictions) < trivial


def test_pipeline_is_fitted_on_training_data_only(small_config, credit_like_data):
    X, y = credit_like_data
    X_train, X_test, y_train, _ = stratified_split(X, y, test_size=0.25, seed=1)
    pipeline = build_pipeline(small_config).fit(X_train, y_train)
    numeric = pipeline.preprocessor_.numeric_columns_
    np.testing.assert_allclose(
        pipeline.preprocessor_.scaler_.mean_, X_train[numeric].mean().to_numpy()
    )
    assert pipeline.svm.alpha_.shape == (len(X_train),)
    # Cambiar el test no cambia el modelo ni las predicciones sobre train.
    before = pipeline.decision_function(X_train)
    pipeline.decision_function(X_test.assign(age=X_test["age"] + 100.0))
    np.testing.assert_array_equal(pipeline.decision_function(X_train), before)


def test_sensitive_attributes_can_be_excluded_by_configuration(small_config, credit_like_data):
    X, y = credit_like_data
    full = build_pipeline(small_config).fit(X, y)
    reduced = build_pipeline(with_overrides(small_config, {"data.exclude_sensitive": True})).fit(
        X, y
    )
    sensitive = set(small_config["data"]["sensitive_features"])
    used = {name.split("=")[0] for name in reduced.preprocessor_.feature_names_}
    assert used.isdisjoint(sensitive)
    assert sensitive <= {name.split("=")[0] for name in full.preprocessor_.feature_names_}
    # El modelo reducido ignora por completo los atributos sensibles.
    shuffled = X.copy()
    shuffled["age"] = shuffled["age"].to_numpy()[::-1]
    shuffled["foreign_worker"] = "no"
    np.testing.assert_array_equal(reduced.predict(shuffled), reduced.predict(X))


def test_pipeline_with_kernel_pca_stage(small_config, credit_like_data):
    X, y = credit_like_data
    config = with_overrides(
        small_config,
        {"kpca.enabled": True, "kpca.n_components": 6, "svm.kernel": {"name": "linear"}},
    )
    pipeline = build_pipeline(config).fit(X, y)
    assert pipeline.transform(X).shape == (len(X), 6)
    assert pipeline.svm.support_vectors_.shape[1] == 6
    assert set(np.unique(pipeline.predict(X))) <= {0, 1}


def test_baselines(small_config, credit_like_data):
    X, y = credit_like_data
    majority = MajorityClassifier().fit(X, y)
    assert np.all(majority.predict(X) == 0)  # la clase mayoritaria es "good"
    assert np.ptp(majority.decision_function(X)) == 0.0
    assert np.all(ConstantClassifier(1).fit(X, y).predict(X) == 1)

    logistic = build_logistic_baseline(small_config).fit(X, y)
    assert logistic.class_weight == {1: 5.0, 0: 1.0}
    assert np.array_equal(logistic.predict(X) == 1, logistic.decision_function(X) > 0)


# ------------------------------------------------------------------ tuning
def test_expand_grid_and_folds(credit_like_data):
    grid = {"svm.C": [1, 10], "svm.kernel.gamma": [0.1, 0.2, 0.3]}
    combinations = expand_grid(grid)
    assert len(combinations) == 6
    assert combinations[0] == {"svm.C": 1, "svm.kernel.gamma": 0.1}

    _, y = credit_like_data
    folds = make_folds(y, n_splits=3, seed=0)
    validation = np.concatenate([valid for _, valid in folds])
    assert sorted(validation) == list(range(len(y)))  # cada fila se valida exactamente una vez
    for train_idx, valid_idx in folds:
        assert set(train_idx).isdisjoint(valid_idx)
        assert abs(y[valid_idx].mean() - y.mean()) < 0.05  # estratificado


def test_grid_search_and_selection(small_config, credit_like_data):
    X, y = credit_like_data
    grid = small_config["cv"]["grid"]
    results = grid_search(small_config, grid, X, y)
    assert len(results) == 4
    assert {"svm.C", "svm.kernel.gamma", "mean_cost", "std_cost", "mean_auc"} <= set(results)
    assert results["n_not_converged"].sum() == 0

    best = select_best(results, grid, "cost")
    assert results["mean_cost"].min() == pytest.approx(
        results[(results["svm.C"] == best["svm.C"])
                & (results["svm.kernel.gamma"] == best["svm.kernel.gamma"])]["mean_cost"].iloc[0]
    )  # fmt: skip
    best_auc = select_best(results, grid, "auc")
    assert set(best_auc) == set(grid)
    assert isinstance(best["svm.C"], float)  # tipo nativo, serializable
    with pytest.raises(ValueError):
        select_best(results, grid, "accuracy")


# ----------------------------------------------------- end-to-end (scripts)
@pytest.fixture
def synthetic_split(credit_like_data, monkeypatch):
    X, y = credit_like_data
    split = stratified_split(X, y, test_size=0.25, seed=0)
    monkeypatch.setattr(train, "load_and_split", lambda config: split)
    monkeypatch.setattr(make_figures, "load_and_split", lambda config: split)
    return split


def test_train_run_writes_metrics_tables_and_latex(small_config, synthetic_split, tmp_path):
    results, tables = train.run(small_config)
    assert set(results["test"]) == {"majority", "always_bad", "logreg", "svm", "kpca_svm"}
    assert results["sklearn_check"]["prediction_agreement"] >= 0.98
    assert results["sklearn_check"]["decision_correlation"] >= 0.999
    assert results["svm_fit"]["converged"]
    assert results["gram"]["condition_number_with_jitter"] <= results["gram"]["condition_number"]
    assert set(results["sensitive"]) == {"age_threshold", "with", "without"}
    assert set(results["sensitive"]["with"]["groups"]) == {"sex", "age_group", "foreign_worker"}
    # La clase mayoritaria tiene la mejor exactitud trivial y el peor costo.
    assert results["test"]["majority"]["cost"] > results["test"]["always_bad"]["cost"]

    results_dir = tmp_path / "results"
    write_outputs(results, tables, results_dir)
    stored = json.loads((results_dir / "metrics.json").read_text())
    assert stored["test"]["svm"]["cost"] == results["test"]["svm"]["cost"]
    for name in (
        "numbers.tex", "table_models.tex", "table_cv.tex", "table_confusion_svm.tex",
        "table_sensitive.tex", "table_subgroups.tex", "cv_svm.csv", "test_predictions.csv",
    ):  # fmt: skip
        assert (results_dir / name).stat().st_size > 0
    macros = (results_dir / "numbers.tex").read_text()
    assert "\\newcommand{\\ResSvmCost}" in macros
    assert "\\newcommand{\\ResKpcaSvmAuc}" in macros


def test_train_and_figures_command_line(small_config, synthetic_split, tmp_path, capsys):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(small_config))
    train.main(["--config", str(config_path), "--quiet"])
    assert "Agreement with sklearn SVC" in capsys.readouterr().out

    make_figures.main(["--config", str(config_path)])
    figures = sorted(path.name for path in (tmp_path / "figures").iterdir())
    assert figures == [
        "cv_heatmap.png",
        "kpca_spectrum.png",
        "projection_kpca_vs_pca.png",
        "roc_curves.png",
        "svm_decision_boundary.png",
    ]


def test_make_figures_requires_training_outputs(small_config, synthetic_split):
    with pytest.raises(FileNotFoundError, match="rkhs_credit.train"):
        make_figures.make_all(small_config)
