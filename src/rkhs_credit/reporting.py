"""Write the results of a training run in the formats the report consumes.

``docs/report.tex`` never contains a hand-typed result: it ``\\input``s the
LaTeX tables and the ``\\Res...`` number macros generated here, so re-running
the training script and recompiling is enough to refresh the document.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

MODEL_LABELS = {
    "majority": "Trivial: clase mayoritaria (aprobar a todos)",
    "always_bad": "Trivial: rechazar a todos",
    "logreg": "Regresión logística",
    "svm": "SVM propia (SMO)",
}
GROUP_ATTRIBUTE_LABELS = {
    "sex": "Sexo",
    "age_group": "Edad",
    "foreign_worker": "Trabajador extranjero",
}


# --------------------------------------------------------------- formatting
def num(value: float, digits: int = 3) -> str:
    """Fixed-point number wrapped in siunitx's ``\\num`` (decimal comma in the PDF)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, float) and math.isinf(value):
        return r"$\infty$"
    return rf"\num{{{value:.{digits}f}}}"


def sci(value: float, digits: int = 2) -> str:
    """Number in scientific notation wrapped in ``\\num``."""
    if isinstance(value, float) and math.isinf(value):
        return r"$\infty$"
    return rf"\num{{{value:.{digits}e}}}"


def integer(value: float) -> str:
    return rf"\num{{{int(round(value))}}}"


def _param_text(params: Mapping) -> str:
    """Human-readable hyper-parameters, e.g. ``$C = 3$, $\\gamma = 0.003$``."""
    symbols = {"C": "C", "gamma": r"\gamma", "n_components": "q", "degree": "d"}
    parts = []
    for dotted_key, value in params.items():
        symbol = symbols.get(dotted_key.split(".")[-1], dotted_key.split(".")[-1])
        parts.append(rf"${symbol} = \num{{{value:g}}}$")
    return ", ".join(parts) if parts else "--"


def _macro_prefix(model_name: str) -> str:
    """``"kpca_svm"`` -> ``"KpcaSvm"`` (LaTeX macro names admit letters only)."""
    return "".join(part.capitalize() for part in model_name.split("_") if part.isalpha())


def _tabular(column_spec: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule", " & ".join(header) + r" \\"]
    lines.append(r"\midrule")
    lines.extend(" & ".join(row) + r" \\" if row else r"\midrule" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------- tables
def models_table(test_metrics: Mapping[str, Mapping], labels: Mapping[str, str]) -> str:
    """Test-set comparison of all models."""
    header = [
        "Modelo", "AUC", r"Recall \emph{bad}", r"Precisión \emph{bad}", "Exactitud",
        "FN", "FP", "Costo", "Costo/solic.",
    ]  # fmt: skip
    rows = []
    for name, metrics in test_metrics.items():
        rows.append(
            [
                labels.get(name, name),
                num(metrics["auc"]),
                num(metrics["recall_bad"]),
                num(metrics["precision_bad"]),
                num(metrics["accuracy"]),
                integer(metrics["fn"]),
                integer(metrics["fp"]),
                integer(metrics["cost"]),
                num(metrics["cost_per_sample"]),
            ]
        )
    return _tabular("lrrrrrrrr", header, rows)


def cv_table(cv_summary: Mapping[str, Mapping], labels: Mapping[str, str]) -> str:
    """Selected hyper-parameters and their cross-validation scores."""
    header = ["Modelo", "Hiperparámetros elegidos", "Costo CV (media $\\pm$ desvío)", "AUC CV"]
    rows = []
    for name, summary in cv_summary.items():
        rows.append(
            [
                labels.get(name, name),
                _param_text(summary["best_params"]),
                rf"{num(summary['mean_cost'])} $\pm$ {num(summary['std_cost'])}",
                num(summary["mean_auc"]),
            ]
        )
    return _tabular("llrr", header, rows)


def confusion_table(metrics: Mapping) -> str:
    """2x2 confusion matrix of one model (rows: true class, columns: prediction)."""
    header = ["", r"Predicho \emph{good}", r"Predicho \emph{bad}"]
    rows = [
        [r"Real \emph{good}", integer(metrics["tn"]), integer(metrics["fp"])],
        [r"Real \emph{bad}", integer(metrics["fn"]), integer(metrics["tp"])],
    ]
    return _tabular("lrr", header, rows)


def sensitive_overall_table(sensitive: Mapping[str, Mapping]) -> str:
    """Model with vs. without sensitive attributes: global metrics and approval gaps."""
    labels = {"with": "Con atributos sensibles", "without": "Sin atributos sensibles"}
    header = [
        "Modelo", "AUC", r"Recall \emph{bad}", r"Precisión \emph{bad}", "Costo",
        r"$\Delta$ sexo", r"$\Delta$ edad", r"$\Delta$ extranjero",
    ]  # fmt: skip
    rows = []
    for key, label in labels.items():
        entry = sensitive[key]
        rows.append(
            [
                label,
                num(entry["test"]["auc"]),
                num(entry["test"]["recall_bad"]),
                num(entry["test"]["precision_bad"]),
                integer(entry["test"]["cost"]),
                num(entry["approval_gap"]["sex"]),
                num(entry["approval_gap"]["age_group"]),
                num(entry["approval_gap"]["foreign_worker"]),
            ]
        )
    return _tabular("lrrrrrrr", header, rows)


def subgroup_table(sensitive: Mapping[str, Mapping]) -> str:
    """Approval rate, recall and false-positive rate per subgroup, for both models."""
    header = [
        "Atributo", "Grupo", "$n$", r"$n_{\text{bad}}$",
        "Aprob.", "Recall", "TFP", "Aprob.", "Recall", "TFP",
    ]  # fmt: skip
    rows: list[list[str]] = []
    attributes = list(sensitive["with"]["groups"])
    for position, attribute in enumerate(attributes):
        if position:
            rows.append([])  # linea separadora entre atributos
        with_groups = sensitive["with"]["groups"][attribute]
        without_groups = sensitive["without"]["groups"][attribute]
        for group, with_row in with_groups.items():
            without_row = without_groups[group]
            rows.append(
                [
                    GROUP_ATTRIBUTE_LABELS.get(attribute, attribute),
                    str(group).replace("<=", r"$\leq$").replace(">", "$>$"),
                    integer(with_row["n"]),
                    integer(with_row["n_bad"]),
                    num(with_row["approval_rate"], 2),
                    num(with_row["recall_bad"], 2),
                    num(with_row["false_positive_rate"], 2),
                    num(without_row["approval_rate"], 2),
                    num(without_row["recall_bad"], 2),
                    num(without_row["false_positive_rate"], 2),
                ]
            )
    body = _tabular("llrrrrrrrr", header, rows)
    # Encabezado de dos niveles: "con" y "sin" atributos sensibles.
    top = (
        r"& & & & \multicolumn{3}{c}{Con atributos sensibles} "
        r"& \multicolumn{3}{c}{Sin atributos sensibles} \\" + "\n"
        r"\cmidrule(lr){5-7} \cmidrule(lr){8-10}" + "\n"
    )
    return body.replace(r"\toprule" + "\n", r"\toprule" + "\n" + top, 1)


# -------------------------------------------------------------------- macros
def number_macros(results: Mapping) -> str:
    """LaTeX ``\\newcommand`` definitions for the numbers quoted in the prose."""
    macros: dict[str, str] = {}
    data = results["data"]
    macros["ResNSamples"] = integer(data["n_samples"])
    macros["ResNTrain"] = integer(data["n_train"])
    macros["ResNTest"] = integer(data["n_test"])
    macros["ResNTestBad"] = integer(data["n_test_bad"])
    macros["ResNTestGood"] = integer(data["n_test_good"])
    macros["ResNFeatures"] = integer(data["n_encoded_features"])
    macros["ResDataSource"] = (
        "OpenML" if data["source"].startswith("openml") else "el archivo original de UCI"
    )
    macros["ResCostFN"] = rf"\num{{{results['cost_matrix']['false_negative']:g}}}"
    macros["ResCostFP"] = rf"\num{{{results['cost_matrix']['false_positive']:g}}}"

    for name, metrics in results["test"].items():
        prefix = _macro_prefix(name)
        macros[f"Res{prefix}Auc"] = num(metrics["auc"])
        macros[f"Res{prefix}Recall"] = num(metrics["recall_bad"])
        macros[f"Res{prefix}Precision"] = num(metrics["precision_bad"])
        macros[f"Res{prefix}Accuracy"] = num(metrics["accuracy"])
        macros[f"Res{prefix}Cost"] = integer(metrics["cost"])
        macros[f"Res{prefix}CostPer"] = num(metrics["cost_per_sample"])
        macros[f"Res{prefix}FN"] = integer(metrics["fn"])
        macros[f"Res{prefix}FP"] = integer(metrics["fp"])
        macros[f"Res{prefix}TP"] = integer(metrics["tp"])
        macros[f"Res{prefix}TN"] = integer(metrics["tn"])

    for name, summary in results["cv"].items():
        prefix = _macro_prefix(name)
        macros[f"Res{prefix}CvCost"] = num(summary["mean_cost"])
        macros[f"Res{prefix}CvCostStd"] = num(summary["std_cost"])
        macros[f"Res{prefix}CvAuc"] = num(summary["mean_auc"])
        macros[f"Res{prefix}Params"] = _param_text(summary["best_params"])

    fit = results["svm_fit"]
    macros["ResNSupport"] = integer(fit["n_support"])
    macros["ResNFreeSupport"] = integer(fit["n_free_support"])
    macros["ResNBoundSupport"] = integer(fit["n_support"] - fit["n_free_support"])
    macros["ResSupportPct"] = num(100.0 * fit["n_support"] / data["n_train"], 1)
    macros["ResSmoIter"] = integer(fit["n_iter"])
    macros["ResKktGap"] = sci(fit["kkt_gap"])
    macros["ResKktTol"] = sci(fit["tol"], 0)
    macros["ResBias"] = num(fit["intercept"])
    macros["ResFitSeconds"] = num(fit["fit_seconds"], 2)

    check = results["sklearn_check"]
    macros["ResSkAgreement"] = num(100.0 * check["prediction_agreement"], 1)
    macros["ResSkCorr"] = num(check["decision_correlation"], 6)
    macros["ResSkMaxDiff"] = sci(check["max_abs_decision_difference"])
    macros["ResSkNSupport"] = integer(check["n_support_sklearn"])

    gram = results["gram"]
    macros["ResGramLambdaMax"] = num(gram["lambda_max"], 1)
    macros["ResGramLambdaMin"] = sci(gram["lambda_min"])
    macros["ResGramRank"] = integer(gram["numerical_rank"])
    macros["ResGramCond"] = sci(gram["condition_number"])
    macros["ResGramCondJitter"] = sci(gram["condition_number_with_jitter"])
    macros["ResGramJitter"] = sci(gram["jitter"], 0)
    macros["ResCenteredRank"] = integer(results["centered_gram"]["numerical_rank"])
    macros["ResCenteredCond"] = sci(results["centered_gram"]["effective_condition_number"])

    if "sensitive" in results:
        for key, prefix in (("with", "Sens"), ("without", "NoSens")):
            entry = results["sensitive"][key]
            macros[f"Res{prefix}Auc"] = num(entry["test"]["auc"])
            macros[f"Res{prefix}Recall"] = num(entry["test"]["recall_bad"])
            macros[f"Res{prefix}Cost"] = integer(entry["test"]["cost"])
            macros[f"Res{prefix}Params"] = _param_text(entry["best_params"])
            for attribute, suffix in (("sex", "Sex"), ("age_group", "Age"),
                                      ("foreign_worker", "Foreign")):  # fmt: skip
                macros[f"Res{prefix}Gap{suffix}"] = num(entry["approval_gap"][attribute])
        macros["ResAgeThreshold"] = rf"\num{{{results['sensitive']['age_threshold']:g}}}"

    macros["ResRuntime"] = num(results["runtime_seconds"], 0)
    lines = ["% Generado por rkhs_credit.train -- no editar a mano."]
    lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- writing
def write_outputs(results: Mapping, tables: Mapping[str, pd.DataFrame], results_dir: Path) -> None:
    """Write ``metrics.json``, the CSV tables and every LaTeX snippet."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, table in tables.items():
        table.to_csv(results_dir / f"{name}.csv", index=name == "test_predictions")

    latex = {
        "table_models.tex": models_table(results["test"], results["labels"]),
        "table_cv.tex": cv_table(results["cv"], results["labels"]),
        "table_confusion_svm.tex": confusion_table(results["test"]["svm"]),
        "table_confusion_logreg.tex": confusion_table(results["test"]["logreg"]),
        "numbers.tex": number_macros(results),
    }
    if "sensitive" in results:
        latex["table_sensitive.tex"] = sensitive_overall_table(results["sensitive"])
        latex["table_subgroups.tex"] = subgroup_table(results["sensitive"])
    for filename, content in latex.items():
        (results_dir / filename).write_text(content, encoding="utf-8")
