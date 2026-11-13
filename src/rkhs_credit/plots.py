"""Figures of the report.

Every function builds one ``matplotlib`` figure from plain arrays and returns
it; :func:`save_figure` writes it to disk.  No function loads data or fits a
model, so they can be reused from a notebook.

Visual conventions
    * Class colours are fixed across figures: "good" is blue with round
      markers, "bad" is orange with triangular markers (the shape is a second
      encoding, so the figures survive greyscale printing and colour-vision
      deficiencies).
    * Magnitudes use a single-hue light-to-dark ramp.
    * No figure uses two different y-scales on the same axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, roc_curve

# ------------------------------------------------------------------ palette
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # azul, naranja, verde agua
SEQUENTIAL_BLUES = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)  # fmt: skip

CLASS_STYLE = {
    0: {"label": "good", "color": SERIES[0], "marker": "o"},
    1: {"label": "bad", "color": SERIES[1], "marker": "^"},
}


def apply_style() -> None:
    """Set the shared matplotlib style (thin marks, recessive axes and grid)."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def save_figure(fig: Figure, stem: str | Path, formats: Sequence[str], dpi: int) -> list[Path]:
    """Write ``fig`` as ``<stem>.<format>`` for every format and close it."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        # Sin fecha de creacion: el archivo no cambia si los datos no cambian.
        metadata = {"CreationDate": None} if extension == "pdf" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def decimal_comma(value: float) -> str:
    """Format a number with a decimal comma (the figures are in Spanish)."""
    return f"{value:g}".replace(".", ",")


def _use_decimal_comma(*axes_list) -> None:
    """Show the tick labels of linear axes with a decimal comma."""
    formatter = FuncFormatter(lambda value, _position: decimal_comma(round(value, 10)))
    for axis in axes_list:
        axis.set_major_formatter(formatter)


def _class_scatter(ax, points: np.ndarray, y: np.ndarray, size: float = 16, alpha: float = 0.8):
    """Scatter of 2-D points with the fixed class colours and marker shapes."""
    # Se dibuja primero la clase mayoritaria para que no tape a la minoritaria.
    for label in (0, 1):
        style = CLASS_STYLE[label]
        mask = y == label
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=size,
            c=style["color"],
            marker=style["marker"],
            alpha=alpha,
            edgecolors=SURFACE,
            linewidths=0.4,
            zorder=2 + label,
        )


def _class_handles() -> list[Line2D]:
    return [
        Line2D(
            [], [], linestyle="none", marker=style["marker"], markersize=6,
            markerfacecolor=style["color"], markeredgecolor=SURFACE, label=style["label"],
        )
        for style in CLASS_STYLE.values()
    ]  # fmt: skip


# ----------------------------------------------------------------- figures
def plot_projection_comparison(
    panels: Mapping[str, tuple[np.ndarray, np.ndarray]], y: np.ndarray
) -> Figure:
    """Side-by-side 2-D projections coloured by class.

    Parameters
    ----------
    panels : mapping {title: (scores, explained_variance_ratio)}
        ``scores`` has shape (n, 2); the ratios label the axes.
    y : array of shape (n,)
        1 = bad, 0 = good.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(7.4, 3.5), constrained_layout=True)
    for ax, (title, (scores, ratios)) in zip(np.atleast_1d(axes), panels.items(), strict=True):
        _class_scatter(ax, scores, y)
        ax.set_title(title)
        ax.set_xlabel(f"Componente 1 ({decimal_comma(round(100 * ratios[0], 1))} % de la varianza)")
        ax.set_ylabel(f"Componente 2 ({decimal_comma(round(100 * ratios[1], 1))} % de la varianza)")
        _use_decimal_comma(ax.xaxis, ax.yaxis)
    np.atleast_1d(axes)[-1].legend(handles=_class_handles(), title="Clase", loc="best")
    return fig


def plot_spectrum(
    spectra: Mapping[str, np.ndarray], n_show: int, variance_marks: Sequence[float] = (0.9,)
) -> Figure:
    """Eigenvalues of centred Gram matrices and cumulative explained variance.

    Parameters
    ----------
    spectra : mapping {label: eigenvalues in decreasing order}
        One curve per entry (at most three).
    n_show : int
        Number of leading components shown.
    variance_marks : sequence of float
        Cumulative-variance levels annotated on the first curve.

    Two panels with a shared x-axis are used instead of one panel with two
    y-scales, which would suggest a relationship between the scales.
    """
    fig, (ax_eig, ax_cum) = plt.subplots(1, 2, figsize=(7.4, 3.3), constrained_layout=True)
    for position, (color, (label, spectrum)) in enumerate(
        zip(SERIES, spectra.items(), strict=False)
    ):
        positive = np.asarray(spectrum, dtype=float)
        positive = positive[positive > 0]
        cumulative = np.cumsum(positive) / positive.sum()
        shown = min(n_show, positive.size)
        index = np.arange(1, shown + 1)
        ax_eig.plot(index, positive[:shown] / positive[0], color=color, label=label)
        ax_cum.plot(index, cumulative[:shown], color=color, label=label)
        if position > 0:
            continue
        # Rotulos selectivos (solo la primera curva): componentes para cada umbral.
        for level in variance_marks:
            needed = int(np.searchsorted(cumulative, level) + 1)
            if needed > shown:
                continue
            ax_cum.plot(
                [needed], [cumulative[needed - 1]], marker="o", markersize=7,
                color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
            )  # fmt: skip
            ax_cum.annotate(
                f"{100 * level:.0f} % de la varianza\ncon {needed} componentes",
                xy=(needed, cumulative[needed - 1]),
                xytext=(0.97, 0.52),
                textcoords="axes fraction",
                ha="right",
                va="top",
                color=INK_SECONDARY,
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.7,
                            "shrinkA": 2, "shrinkB": 5},
            )  # fmt: skip
    ax_eig.set_yscale("log")
    ax_eig.set_title("Autovalores de $\\tilde{K}$")
    ax_eig.set_xlabel("Componente $k$")
    ax_eig.set_ylabel("$\\lambda_k / \\lambda_1$ (escala logarítmica)")
    ax_cum.set_ylim(0, 1.03)
    _use_decimal_comma(ax_cum.yaxis)
    ax_cum.set_title("Varianza explicada acumulada")
    ax_cum.set_xlabel("Número de componentes $q$")
    ax_cum.set_ylabel("$\\sum_{k \\leq q} \\lambda_k \\,/\\, \\mathrm{tr}\\,\\tilde{K}$")
    if len(spectra) > 1:
        ax_cum.legend(loc="lower right")
    return fig


def plot_decision_boundary(
    scores: np.ndarray,
    y: np.ndarray,
    decision_function,
    support_indices: np.ndarray,
    free_support_indices: np.ndarray,
    grid_size: int,
) -> Figure:
    """SVM decision regions on the first two Kernel PCA components.

    Parameters
    ----------
    scores : array of shape (n, 2)
        Training points in the Kernel PCA plane.
    decision_function : callable
        Maps an array of shape (m, 2) to the SVM scores ``f``.
    support_indices : array of int
        Rows of ``scores`` that are support vectors (``alpha_i > 0``).
    free_support_indices : array of int
        Support vectors with ``0 < alpha_i < C_i``; they lie on the margins.
    """
    margin = 0.06 * np.ptp(scores, axis=0)
    x_min, y_min = scores.min(axis=0) - margin
    x_max, y_max = scores.max(axis=0) + margin
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size)
    )
    values = decision_function(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    values = values.reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    ax.grid(False)
    # Regiones de decision como un lavado tenue del color de la clase predicha.
    region_colors = [CLASS_STYLE[0]["color"], CLASS_STYLE[1]["color"]]
    ax.contourf(
        grid_x, grid_y, values, levels=[-np.inf, 0, np.inf],
        colors=region_colors, alpha=0.13, zorder=0,
    )  # fmt: skip
    # Frontera f = 0 (linea llena) y margenes f = +-1 (linea de trazos).
    ax.contour(grid_x, grid_y, values, levels=[0], colors=INK, linewidths=1.8, zorder=1)
    ax.contour(
        grid_x, grid_y, values, levels=[-1, 1], colors=INK_SECONDARY,
        linewidths=0.9, linestyles="dashed", zorder=1,
    )  # fmt: skip

    # Puntos con alpha = 0 (no son vectores de soporte): tenues.
    is_support = np.zeros(scores.shape[0], dtype=bool)
    is_support[support_indices] = True
    _class_scatter(ax, scores[~is_support], y[~is_support], size=12, alpha=0.3)
    # Vectores de soporte: opacos.
    _class_scatter(ax, scores[is_support], y[is_support], size=18, alpha=0.9)
    # Vectores de soporte libres (0 < alpha < C_i): anillo; caen sobre los margenes.
    ax.scatter(
        scores[free_support_indices, 0], scores[free_support_indices, 1],
        s=52, facecolors="none", edgecolors=INK, linewidths=0.7, zorder=5,
    )  # fmt: skip
    ax.set_xlabel("Componente 1 de Kernel PCA")
    ax.set_ylabel("Componente 2 de Kernel PCA")
    _use_decimal_comma(ax.xaxis, ax.yaxis)
    handles = [
        *_class_handles(),
        Line2D(
            [], [], linestyle="none", marker="o", markersize=5, markerfacecolor=INK_MUTED,
            markeredgecolor=SURFACE, alpha=0.35, label="no es vector de soporte ($\\alpha_i = 0$)",
        ),
        Line2D(
            [], [], linestyle="none", marker="o", markersize=6, markerfacecolor=INK_MUTED,
            markeredgecolor=SURFACE, label="vector de soporte en la cota ($\\alpha_i = C_i$)",
        ),
        Line2D(
            [], [], linestyle="none", marker="o", markersize=8, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=0.7,
            label="vector de soporte libre ($0 < \\alpha_i < C_i$)",
        ),
        Line2D([], [], color=INK, linewidth=1.8, label="frontera $f(x) = 0$"),
        Line2D(
            [], [], color=INK_SECONDARY, linewidth=0.9, linestyle="dashed",
            label="márgenes $f(x) = \\pm 1$",
        ),
        Rectangle((0, 0), 1, 1, facecolor=region_colors[0], alpha=0.25,
                  label="región clasificada good"),
        Rectangle((0, 0), 1, 1, facecolor=region_colors[1], alpha=0.25,
                  label="región clasificada bad"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    scores_by_model: Mapping[str, np.ndarray],
    predictions_by_model: Mapping[str, np.ndarray],
) -> Figure:
    """ROC curves of up to three models plus the trivial (chance) diagonal.

    The marker on each curve is the operating point of the model's hard
    predictions (the threshold actually used to classify).
    """
    if len(scores_by_model) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} models can be drawn with distinct colours.")
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    ax.plot(
        [0, 1], [0, 1], color=INK_MUTED, linewidth=1.2,
        label="Clasificador trivial (AUC = 0,500)",
    )  # fmt: skip
    for color, (name, scores) in zip(SERIES, scores_by_model.items(), strict=False):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = f"{roc_auc_score(y_true, scores):.3f}".replace(".", ",")
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC = {auc})")
        predictions = np.asarray(predictions_by_model[name])
        point_fpr = np.mean(predictions[y_true == 0] == 1)
        point_tpr = np.mean(predictions[y_true == 1] == 1)
        ax.plot(
            [point_fpr], [point_tpr], linestyle="none", marker="o", markersize=8,
            color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
        )  # fmt: skip
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    _use_decimal_comma(ax.xaxis, ax.yaxis)
    ax.set_xlabel("Tasa de falsos positivos (good clasificados bad)")
    ax.set_ylabel("Recall de la clase bad")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [], [], linestyle="none", marker="o", markersize=7, markerfacecolor=INK_MUTED,
            markeredgecolor=SURFACE, label="punto de operación del modelo",
        )
    )  # fmt: skip
    ax.legend(handles=handles, loc="lower right")
    return fig


def plot_cv_heatmap(
    cv_results: pd.DataFrame,
    row_key: str,
    column_key: str,
    best_params: Mapping,
    value_key: str = "mean_cost",
    row_label: str = "$C$",
    column_label: str = "$\\gamma$",
    value_label: str = "Costo medio por solicitante (validación cruzada)",
) -> Figure:
    """Heat map of a cross-validation score over a two-parameter grid."""
    table = cv_results.pivot(index=row_key, columns=column_key, values=value_key).sort_index()
    values = table.to_numpy()
    cmap = LinearSegmentedColormap.from_list("blues", SEQUENTIAL_BLUES)

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    ax.grid(False)
    n_rows, n_columns = values.shape
    # Celdas separadas por un filete del color del fondo (sin bordes dibujados).
    image = ax.pcolormesh(
        np.arange(n_columns + 1) - 0.5, np.arange(n_rows + 1) - 0.5, values,
        cmap=cmap, edgecolors=SURFACE, linewidth=1.5,
    )  # fmt: skip
    ax.set_xticks(range(n_columns), [decimal_comma(value) for value in table.columns])
    ax.set_yticks(range(n_rows), [decimal_comma(value) for value in table.index])
    ax.set_xlabel(column_label)
    ax.set_ylabel(row_label)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Valor de cada celda (hace las veces de tabla); la tinta se elige segun la
    # luminancia del color de fondo de la celda.
    for i in range(n_rows):
        for j in range(n_columns):
            red, green, blue, _ = cmap(image.norm(values[i, j]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j, i, f"{values[i, j]:.3f}".replace(".", ","), ha="center", va="center",
                fontsize=7, color=INK if luminance > 0.5 else SURFACE,
            )  # fmt: skip
    # Celda elegida por la validacion cruzada.
    best_row = list(table.index).index(best_params[row_key])
    best_column = list(table.columns).index(best_params[column_key])
    ax.add_patch(
        Rectangle((best_column - 0.5, best_row - 0.5), 1, 1, fill=False,
                  edgecolor=SERIES[1], linewidth=2.2)
    )  # fmt: skip
    ax.legend(
        handles=[Rectangle((0, 0), 1, 1, fill=False, edgecolor=SERIES[1], linewidth=2.2,
                           label="combinación elegida")],
        loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0,
    )  # fmt: skip
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(value_label)
    colorbar.outline.set_visible(False)
    _use_decimal_comma(colorbar.ax.yaxis)
    return fig
