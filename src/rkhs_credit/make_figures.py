"""Generate every figure of the report into ``paths.figures_dir``.

Usage::

    python -m rkhs_credit.make_figures --config configs/default.yaml

The ROC curves and the cross-validation heat map are drawn from the files
written by ``rkhs_credit.train`` (run it first).  The Kernel PCA figures are
recomputed here from the cached data, which takes a couple of seconds.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el script solo escribe archivos

import pandas as pd  # noqa: E402

from rkhs_credit import plots  # noqa: E402
from rkhs_credit.config import load_config  # noqa: E402
from rkhs_credit.data import TabularPreprocessor, load_and_split  # noqa: E402
from rkhs_credit.evaluation import CostMatrix  # noqa: E402
from rkhs_credit.kernels import LinearKernel, RBFKernel  # noqa: E402
from rkhs_credit.kpca import KernelPCA  # noqa: E402
from rkhs_credit.pipeline import excluded_columns, resolve_class_weight  # noqa: E402
from rkhs_credit.svm import KernelSVM  # noqa: E402

ROC_MODELS = ("svm", "logreg", "kpca_svm")


def _load_training_outputs(results_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    required = [
        results_dir / name for name in ("metrics.json", "cv_svm.csv", "test_predictions.csv")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing training outputs {missing}. Run `python -m rkhs_credit.train` first."
        )
    metrics = json.loads(required[0].read_text(encoding="utf-8"))
    return metrics, pd.read_csv(required[1]), pd.read_csv(required[2], index_col="row")


def _axis_label(dotted_key: str) -> str:
    """Axis label of a grid parameter, e.g. ``"svm.kernel.gamma"`` -> ``$\\gamma$``."""
    name = dotted_key.split(".")[-1]
    return {"C": "$C$", "gamma": "$\\gamma$"}.get(name, name)


def make_all(config: Mapping) -> list[Path]:
    """Create all figures and return the paths written."""
    figures_cfg = config["figures"]
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics, cv_results, predictions = _load_training_outputs(Path(config["paths"]["results_dir"]))
    best_params = metrics["cv"]["svm"]["best_params"]
    best_gamma = best_params.get("svm.kernel.gamma", config["svm"]["kernel"].get("gamma"))
    best_C = best_params.get("svm.C", config["svm"]["C"])

    plots.apply_style()
    written: list[Path] = []

    def save(fig, name: str) -> None:
        written.extend(
            plots.save_figure(fig, figures_dir / name, figures_cfg["formats"], figures_cfg["dpi"])
        )

    # Datos de train preprocesados (el preprocesador se ajusta solo con train).
    X_train, _, y_train, _ = load_and_split(config)
    features = TabularPreprocessor(exclude=excluded_columns(config)).fit_transform(X_train)

    # ---- 1. Kernel PCA (RBF) vs PCA lineal --------------------------------
    gamma = figures_cfg.get("projection_gamma") or best_gamma
    kpca_rbf = KernelPCA(RBFKernel(gamma), eig_rtol=config["kpca"]["eig_rtol"])
    scores_rbf = kpca_rbf.fit_transform(features)[:, :2]
    # PCA lineal = Kernel PCA con el kernel lineal.
    pca_linear = KernelPCA(LinearKernel(), eig_rtol=config["kpca"]["eig_rtol"])
    scores_linear = pca_linear.fit_transform(features)[:, :2]
    save(
        plots.plot_projection_comparison(
            {
                f"Kernel PCA (RBF, $\\gamma$ = {plots.decimal_comma(gamma)})": (
                    scores_rbf,
                    kpca_rbf.explained_variance_ratio_[:2],
                ),
                "PCA lineal": (scores_linear, pca_linear.explained_variance_ratio_[:2]),
            },
            y_train,
        ),
        "projection_kpca_vs_pca",
    )

    # ---- 2. Espectro de la matriz de Gram centrada -------------------------
    save(
        plots.plot_spectrum(
            {
                f"Kernel PCA (RBF, $\\gamma$ = {plots.decimal_comma(gamma)})": kpca_rbf.spectrum_,
                "PCA lineal": pca_linear.spectrum_,
            },
            figures_cfg["spectrum_components"],
        ),
        "kpca_spectrum",
    )

    # ---- 3. Frontera de decision sobre las 2 primeras componentes ----------
    cost = CostMatrix.from_config(config)
    svm_2d = KernelSVM(
        kernel=RBFKernel(figures_cfg["boundary_gamma"]),
        C=figures_cfg.get("boundary_C") or best_C,
        class_weight=resolve_class_weight(config["svm"].get("class_weight"), cost),
        tol=config["svm"]["tol"],
        max_iter=config["svm"]["max_iter"],
        jitter=config["svm"].get("jitter", 0.0),
    ).fit(scores_rbf, y_train)
    save(
        plots.plot_decision_boundary(
            scores_rbf,
            y_train,
            svm_2d.decision_function,
            svm_2d.support_,
            svm_2d.free_support_,
            figures_cfg["boundary_grid_size"],
        ),
        "svm_decision_boundary",
    )

    # ---- 4. Curvas ROC en test ---------------------------------------------
    labels = metrics["labels"]
    available = [name for name in ROC_MODELS if f"{name}_score" in predictions.columns]
    save(
        plots.plot_roc_curves(
            predictions["y_true"].to_numpy(),
            {labels[name]: predictions[f"{name}_score"].to_numpy() for name in available},
            {labels[name]: predictions[f"{name}_prediction"].to_numpy() for name in available},
        ),
        "roc_curves",
    )

    # ---- 5. Mapa de calor de la validacion cruzada -------------------------
    grid_keys = list(config["cv"]["grid"])
    if len(grid_keys) == 2 and all(key in cv_results.columns for key in grid_keys):
        save(
            plots.plot_cv_heatmap(
                cv_results,
                grid_keys[0],
                grid_keys[1],
                best_params,
                row_label=_axis_label(grid_keys[0]),
                column_label=_axis_label(grid_keys[1]),
            ),
            "cv_heatmap",
        )
    else:
        print("cv_heatmap skipped: it needs a cross-validation grid over exactly two parameters.")
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the YAML file.")
    args = parser.parse_args(argv)
    for path in make_all(load_config(args.config)):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
