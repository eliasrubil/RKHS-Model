"""Soft-margin kernel SVM trained in the dual with SMO.

The module has two layers:

* :func:`smo` is the numerical routine.  It receives a Gram matrix, labels in
  ``{-1, +1}`` and per-sample upper bounds, and returns the dual solution.  It
  knows nothing about kernels or class labels.
* :class:`KernelSVM` is the estimator: it encodes the labels, turns class
  weights into per-sample bounds, calls :func:`smo`, computes the bias from
  the support vectors and exposes ``decision_function`` / ``predict``.

Algorithm
---------
Each iteration optimises two multipliers analytically, exactly as in Platt's
SMO (1998).  The pair is chosen with the *maximal violating pair* rule of
Keerthi et al. (2001) instead of Platt's original heuristics: it is
deterministic, the stopping rule is literally "the KKT conditions hold up to
``tol``", and the bias is recovered at the end instead of being tracked.
The full Gram matrix is kept in memory (O(n^2)), which is fine for the
~10^3 samples of this project; see "Limitaciones" in the report.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from rkhs_credit.kernels import Kernel
from rkhs_credit.numerics import add_jitter, as_2d_float

# Piso para la curvatura eta cuando dos puntos coinciden en el espacio de
# caracteristicas (eta = 0): ver Ec. (A.40) [eq:smo-eta].
ETA_FLOOR = 1e-12
# Tolerancia relativa para "pegar" un multiplicador a su cota (0 o C_i) y para
# decidir que un multiplicador es positivo (vector de soporte).
BOUND_RTOL = 1e-10


@dataclass
class SMOResult:
    """Output of :func:`smo`."""

    alpha: np.ndarray  # multiplicadores duales, forma (n,)
    errors: np.ndarray  # E_t = sum_j alpha_j y_j K_tj - y_t (sin sesgo), forma (n,)
    n_iter: int
    converged: bool
    kkt_gap: float  # m(alpha) - M(alpha); <= tol si convergio
    b_low: float  # m(alpha) = max_{I_up} (-E_t): cota inferior para b
    b_high: float  # M(alpha) = min_{I_low} (-E_t): cota superior para b


def _index_sets(alpha: np.ndarray, positive: np.ndarray, upper: np.ndarray):
    """Boolean masks of the index sets I_up and I_low."""
    # I_up  = {t : (y_t = +1, alpha_t < C_t)  o  (y_t = -1, alpha_t > 0)}
    # I_low = {t : (y_t = +1, alpha_t > 0)    o  (y_t = -1, alpha_t < C_t)}
    # Ec. (A.33) [eq:index-sets].
    below_upper = alpha < upper
    above_zero = alpha > 0
    up = np.where(positive, below_upper, above_zero)
    low = np.where(positive, above_zero, below_upper)
    return up, low


def _violation(errors: np.ndarray, up: np.ndarray, low: np.ndarray):
    """Maximal violating pair (i, j) and the bounds m(alpha), M(alpha)."""
    # i = argmax_{t en I_up} (-E_t),  j = argmin_{t en I_low} (-E_t):
    # Ec. (A.47) [eq:smo-working-set].
    minus_e = -errors
    i = int(np.argmax(np.where(up, minus_e, -np.inf)))
    j = int(np.argmin(np.where(low, minus_e, np.inf)))
    return i, j, float(minus_e[i]), float(minus_e[j])


def smo(
    K: np.ndarray,
    y: np.ndarray,
    upper: np.ndarray | float,
    tol: float = 1e-3,
    max_iter: int = 100_000,
) -> SMOResult:
    """Solve the SVM dual with Sequential Minimal Optimization.

    The problem solved is (Ec. (A.26) [eq:svm-dual])::

        max_alpha  1^T alpha - 1/2 alpha^T Q alpha,   Q_ij = y_i y_j K_ij
        s.t.       0 <= alpha_i <= upper_i,   y^T alpha = 0.

    Parameters
    ----------
    K : ndarray of shape (n, n)
        Symmetric positive semidefinite Gram matrix.
    y : ndarray of shape (n,)
        Labels in ``{-1, +1}``.
    upper : float or ndarray of shape (n,)
        Box bounds ``C_i`` (one per sample; this is how class weights enter).
    tol : float
        KKT tolerance: stop when ``m(alpha) - M(alpha) <= tol``.
    max_iter : int
        Maximum number of two-multiplier updates.
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    K = np.asarray(K, dtype=float)
    if K.shape != (n, n):
        raise ValueError(f"K must have shape ({n}, {n}); got {K.shape}.")
    if not np.all(np.isin(y, (-1.0, 1.0))):
        raise ValueError("y must contain only -1 and +1.")
    upper = np.broadcast_to(np.asarray(upper, dtype=float), (n,)).copy()
    if np.any(upper <= 0):
        raise ValueError("All upper bounds C_i must be positive.")

    positive = y > 0
    alpha = np.zeros(n)
    # Cache de errores sin sesgo, E_t = sum_j alpha_j y_j K_tj - y_t,
    # Ec. (A.32) [eq:error-cache]. Con alpha = 0 queda E = -y.
    errors = -y.copy()

    n_iter = 0
    converged = False
    while True:
        up, low = _index_sets(alpha, positive, upper)
        i, j, b_low, b_high = _violation(errors, up, low)
        # Las condiciones KKT equivalen a m(alpha) <= b <= M(alpha),
        # Ec. (A.34) [eq:kkt-interval]; se detiene cuando se cumplen salvo tol,
        # Ec. (A.48) [eq:smo-stopping].
        if b_low - b_high <= tol:
            converged = True
            break
        if n_iter >= max_iter:
            break

        # --- Paso analitico sobre el par (i, j) -------------------------
        # Curvatura a lo largo de la recta factible:
        # eta = K_ii + K_jj - 2 K_ij = ||phi(x_i) - phi(x_j)||^2 >= 0,
        # Ec. (A.40) [eq:smo-eta].
        eta = max(K[i, i] + K[j, j] - 2.0 * K[i, j], ETA_FLOOR)
        s = y[i] * y[j]
        # Maximo sin restricciones de la parabola: Ec. (A.41) [eq:smo-step].
        alpha_j_new = alpha[j] + y[j] * (errors[i] - errors[j]) / eta
        # Extremos [L, H] del segmento factible dentro de la caja:
        if s < 0:  # y_i != y_j, Ec. (A.42) [eq:smo-box-opposite]
            low_end = max(0.0, alpha[j] - alpha[i])
            high_end = min(upper[j], upper[i] + alpha[j] - alpha[i])
        else:  # y_i == y_j, Ec. (A.43) [eq:smo-box-same]
            low_end = max(0.0, alpha[i] + alpha[j] - upper[i])
            high_end = min(upper[j], alpha[i] + alpha[j])
        # Recorte a la caja: Ec. (A.44) [eq:smo-clip].
        alpha_j_new = min(max(alpha_j_new, low_end), high_end)
        # alpha_i se mueve para conservar y†alpha = 0: Ec. (A.45) [eq:smo-alpha-i].
        alpha_i_new = alpha[i] + s * (alpha[j] - alpha_j_new)
        alpha_i_new = _snap_to_bounds(alpha_i_new, upper[i])
        alpha_j_new = _snap_to_bounds(alpha_j_new, upper[j])

        # Actualizacion O(n) del cache de errores: Ec. (A.46) [eq:smo-error-update].
        errors += y[i] * (alpha_i_new - alpha[i]) * K[i] + y[j] * (alpha_j_new - alpha[j]) * K[j]
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new
        n_iter += 1

    return SMOResult(
        alpha=alpha,
        errors=errors,
        n_iter=n_iter,
        converged=converged,
        kkt_gap=b_low - b_high,
        b_low=b_low,
        b_high=b_high,
    )


def _snap_to_bounds(value: float, upper: float) -> float:
    """Clip to ``[0, upper]`` and remove round-off residue next to the bounds.

    Without this, a multiplier that should be exactly ``0`` or ``C_i`` may be
    left at ``1e-17`` or ``C_i - 1e-17`` and be treated as a free variable.
    """
    if value <= BOUND_RTOL * upper:
        return 0.0
    if value >= upper * (1.0 - BOUND_RTOL):
        return float(upper)
    return float(value)


class KernelSVM:
    """Binary soft-margin SVM with an arbitrary :class:`~rkhs_credit.kernels.Kernel`.

    Parameters
    ----------
    kernel : Kernel
        Positive-definite kernel.
    C : float
        Regularisation constant (larger = less regularisation).
    class_weight : None, "balanced" or mapping {label: weight}
        Sample ``i`` gets the box bound ``C_i = C * weight[y_i]``; this is how
        an asymmetric misclassification cost enters the training problem.
    tol : float
        KKT tolerance of the SMO stopping rule.
    max_iter : int
        Maximum number of SMO iterations; a ``RuntimeWarning`` is emitted when
        it is reached without convergence.
    jitter : float
        Value added to the diagonal of the *training* Gram matrix.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Sorted labels; ``classes_[1]`` is the positive class (as in scikit-learn).
    alpha_ : ndarray of shape (n_train,)
        Dual variables.
    support_ : ndarray of int
        Indices of the support vectors (``alpha_i > 0``).
    support_vectors_ : ndarray of shape (n_sv, d)
    dual_coef_ : ndarray of shape (n_sv,)
        ``alpha_i * y_i`` for the support vectors.
    intercept_ : float
        Bias ``b``.
    free_support_ : ndarray of int
        Indices of the *free* support vectors (``0 < alpha_i < C_i``), which lie
        exactly on the margin.  ``n_free_support_`` is their number.
    n_iter_, converged_, kkt_gap_ : SMO diagnostics.
    dual_objective_ : float
        Value of the dual objective at the solution.
    """

    def __init__(
        self,
        kernel: Kernel,
        C: float = 1.0,
        class_weight: Mapping | str | None = None,
        tol: float = 1e-3,
        max_iter: int = 100_000,
        jitter: float = 0.0,
    ):
        if C <= 0:
            raise ValueError("C must be positive.")
        self.kernel = kernel
        self.C = float(C)
        self.class_weight = class_weight
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.jitter = float(jitter)

    # ---------------------------------------------------------------- fit
    def fit(self, X, y) -> KernelSVM:
        X = as_2d_float(X, "X")
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have inconsistent numbers of samples.")
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(f"KernelSVM is a binary classifier; got classes {self.classes_}.")
        y_signed = np.where(y == self.classes_[1], 1.0, -1.0)

        # Pesos por clase: C_i = C * w_{y_i}, Ec. (A.35) [eq:class-weights].
        upper = self.C * self._sample_weights(y)

        K = self.kernel.gram(X)
        if self.jitter > 0:
            # K + eps I: vuelve estrictamente concavo cada subproblema de dos
            # variables (eta >= 2 eps > 0), Ec. (A.80) [eq:jitter-eta].
            K = add_jitter(K, self.jitter)

        result = smo(K, y_signed, upper, tol=self.tol, max_iter=self.max_iter)
        if not result.converged:
            warnings.warn(
                f"SMO stopped after max_iter={self.max_iter} iterations with KKT gap "
                f"{result.kkt_gap:.3e} > tol={self.tol:.1e}. Increase max_iter or rescale "
                "the data / lower C.",
                RuntimeWarning,
                stacklevel=2,
            )

        alpha = result.alpha
        # Vectores de soporte: alpha_i > 0. Solo ellos aparecen en f,
        # Ec. (A.29) [eq:decision-function].
        is_support = alpha > BOUND_RTOL * upper
        # Vectores de soporte libres (0 < alpha_i < C_i): estan exactamente sobre
        # el margen, y_i f(x_i) = 1, Ec. (A.30) [eq:kkt-cases].
        is_free = is_support & (alpha < upper * (1.0 - BOUND_RTOL))
        if np.any(is_free):
            # Para cada vector libre, b = y_i - sum_j alpha_j y_j K_ij = -E_i,
            # Ec. (A.31) [eq:bias-free-sv]; se promedia para reducir el error numerico.
            intercept = float(np.mean(-result.errors[is_free]))
        else:
            # Sin vectores libres, b solo queda acotado al intervalo [m, M] de
            # Ec. (A.34) [eq:kkt-interval]; se toma el punto medio.
            intercept = 0.5 * (result.b_low + result.b_high)

        self.alpha_ = alpha
        self.support_ = np.flatnonzero(is_support)
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = alpha[self.support_] * y_signed[self.support_]
        self.intercept_ = intercept
        self.free_support_ = np.flatnonzero(is_free)
        self.n_free_support_ = int(is_free.sum())
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.kkt_gap_ = result.kkt_gap
        # W(alpha) = 1†alpha - (1/2) alpha†Q alpha, con (Q alpha)_t = y_t (E_t + y_t).
        q_alpha = y_signed * (result.errors + y_signed)
        self.dual_objective_ = float(alpha.sum() - 0.5 * alpha @ q_alpha)
        return self

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Per-sample weight ``w_{y_i}`` derived from ``class_weight``."""
        if self.class_weight is None:
            return np.ones(y.shape[0])
        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError("class_weight must be None, 'balanced' or a mapping.")
            # Misma convencion que scikit-learn: n / (n_clases * n_k).
            counts = np.array([(y == label).sum() for label in self.classes_])
            weights = y.shape[0] / (2.0 * counts)
            mapping = dict(zip(self.classes_.tolist(), weights.tolist(), strict=True))
        else:
            mapping = dict(self.class_weight)
        missing = [label for label in self.classes_.tolist() if label not in mapping]
        if missing:
            raise ValueError(f"class_weight has no entry for classes {missing}.")
        sample_weights = np.array([mapping[label] for label in y.tolist()], dtype=float)
        if np.any(sample_weights <= 0):
            raise ValueError("Class weights must be positive.")
        return sample_weights

    # ------------------------------------------------------------ predict
    def decision_function(self, X) -> np.ndarray:
        """Signed score ``f(x) = sum_i alpha_i y_i k(x_i, x) + b``."""
        self._check_is_fitted()
        X = as_2d_float(X, "X")
        if self.support_.size == 0:
            return np.full(X.shape[0], self.intercept_)
        # Ec. (A.29) [eq:decision-function]: solo intervienen los vectores de soporte.
        return self.kernel.gram(X, self.support_vectors_) @ self.dual_coef_ + self.intercept_

    def predict(self, X) -> np.ndarray:
        """Predicted labels: ``classes_[1]`` where ``f(x) > 0``, else ``classes_[0]``."""
        scores = self.decision_function(X)
        return np.where(scores > 0, self.classes_[1], self.classes_[0])

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "alpha_"):
            raise RuntimeError("This KernelSVM instance is not fitted yet; call fit first.")
