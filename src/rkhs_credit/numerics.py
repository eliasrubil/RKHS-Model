"""Numerical-stability helpers shared by the kernel, SVM and Kernel PCA code.

Everything here operates on plain ``numpy`` arrays so that it can be unit
tested in isolation.  The functions are deliberately tiny: each one implements
exactly one of the safeguards discussed in the appendix (section
"Estabilidad numerica") of ``docs/report.tex``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy import linalg

# Tolerancia relativa por defecto para decidir que un autovalor es "cero".
# Los autovalores de una matriz simetrica se calculan con error absoluto
# O(eps_maq * lambda_max): ver Ec. (A.??) [eq:weyl] y Ec. (A.??) [eq:eig-threshold].
DEFAULT_EIG_RTOL = 1e-10


def as_2d_float(X, name: str = "X") -> np.ndarray:
    """Return ``X`` as a 2-D float array of shape (n_samples, n_features)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array of shape (n_samples, n_features); got ndim={X.ndim}."
        )
    return X


def symmetrize(K: np.ndarray) -> np.ndarray:
    """Return the symmetric part ``(K + K^T) / 2`` of a square matrix.

    Floating-point evaluation of a symmetric kernel can produce a matrix that
    differs from its transpose in the last bits; ``scipy.linalg.eigh`` and the
    SMO updates assume exact symmetry.
    """
    # (K + K†)/2 es la matriz simetrica mas cercana a K en norma de Frobenius:
    # Ec. (A.??) [eq:symmetrization].
    K = np.asarray(K, dtype=float)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"symmetrize expects a square matrix; got shape {K.shape}.")
    return 0.5 * (K + K.T)


def add_jitter(K: np.ndarray, jitter: float) -> np.ndarray:
    """Return ``K + jitter * I`` (a copy; ``K`` is left untouched)."""
    # Sumar eps*I desplaza todos los autovalores en eps sin cambiar los
    # autovectores, y acota el numero de condicion: Ec. (A.??) [eq:jitter-cond].
    if jitter < 0:
        raise ValueError("jitter must be non-negative.")
    K = np.array(K, dtype=float, copy=True)
    if jitter > 0:
        K[np.diag_indices_from(K)] += jitter
    return K


def center_gram(K: np.ndarray) -> np.ndarray:
    """Center a training Gram matrix in feature space: ``H K H``.

    ``H = I - (1/n) 1 1^T`` is never formed explicitly; the product is expanded
    into row/column/grand means, which costs O(n^2) instead of O(n^3).
    """
    # K~ = H K H = K - (1/n) 11†K - (1/n) K 11† + (1/n^2) 11†K 11†:
    # Ec. (A.??) [eq:centered-entries] y Ec. (A.??) [eq:centering].
    K = np.asarray(K, dtype=float)
    row_means = K.mean(axis=1, keepdims=True)  # (1/n) K 1
    col_means = K.mean(axis=0, keepdims=True)  # (1/n) 1†K
    grand_mean = K.mean()  # (1/n^2) 1†K 1
    K_centered = K - row_means - col_means + grand_mean
    # El centrado preserva la simetria solo salvo redondeo: se vuelve a simetrizar.
    return symmetrize(K_centered)


def center_gram_test(K_test: np.ndarray, K_train: np.ndarray) -> np.ndarray:
    """Center a test-vs-train kernel matrix using the *training* feature mean.

    Parameters
    ----------
    K_test : array of shape (m, n)
        ``K_test[a, i] = k(x_a_new, x_i_train)``.
    K_train : array of shape (n, n)
        Uncentered training Gram matrix.
    """
    # K~_test = K_test - (1/n) 1_m 1_n†K - (1/n) K_test 1_n 1_n† + (1/n^2) 1_m 1_n†K 1_n 1_n†:
    # Ec. (A.??) [eq:centering-test]. La media que se resta es siempre la de train.
    K_test = np.asarray(K_test, dtype=float)
    K_train = np.asarray(K_train, dtype=float)
    train_col_means = K_train.mean(axis=0, keepdims=True)  # (1/n) 1_n†K, forma (1, n)
    train_grand_mean = K_train.mean()
    test_row_means = K_test.mean(axis=1, keepdims=True)  # (1/n) K_test 1_n, forma (m, 1)
    return K_test - train_col_means - test_row_means + train_grand_mean


def clip_spectrum(
    eigenvalues: np.ndarray, rtol: float = DEFAULT_EIG_RTOL, warn: bool = True
) -> np.ndarray:
    """Set to zero the eigenvalues that are indistinguishable from round-off.

    Eigenvalues with ``|lambda| <= rtol * lambda_max`` become exactly ``0``.
    A *significantly* negative eigenvalue means the matrix is not positive
    semidefinite (i.e. the function is not a valid kernel); it is also clipped
    to zero but a warning is emitted.
    """
    # Umbral relativo al mayor autovalor: Ec. (A.??) [eq:eig-threshold].
    eigenvalues = np.array(eigenvalues, dtype=float, copy=True)
    if eigenvalues.size == 0:
        return eigenvalues
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    threshold = rtol * scale
    significantly_negative = eigenvalues < -threshold
    if warn and np.any(significantly_negative):
        warnings.warn(
            f"{int(significantly_negative.sum())} eigenvalue(s) are significantly negative "
            f"(min = {eigenvalues.min():.3e}); the matrix is not positive semidefinite.",
            RuntimeWarning,
            stacklevel=2,
        )
    eigenvalues[eigenvalues <= threshold] = 0.0
    return eigenvalues


@dataclass(frozen=True)
class SpectralSummary:
    """Spectral diagnostics of a symmetric (ideally PSD) matrix."""

    n: int
    lambda_max: float
    lambda_min: float
    numerical_rank: int
    condition_number: float
    effective_condition_number: float
    jitter: float
    condition_number_with_jitter: float

    def to_dict(self) -> dict:
        return asdict(self)


def spectral_summary(
    K: np.ndarray, jitter: float = 0.0, rtol: float = DEFAULT_EIG_RTOL
) -> SpectralSummary:
    """Compute eigenvalue-based diagnostics, including the condition number.

    * ``condition_number``: ``lambda_max / lambda_min`` of ``K`` itself
      (``inf`` when ``K`` is numerically singular).
    * ``effective_condition_number``: ``lambda_max / (smallest eigenvalue above
      the round-off threshold)``; it is the relevant figure when the null
      space is discarded, as Kernel PCA does.
    * ``condition_number_with_jitter``: condition number of ``K + jitter * I``.
    """
    eigenvalues = linalg.eigvalsh(symmetrize(K))
    # kappa_2(K) = lambda_max / lambda_min para K simetrica definida positiva:
    # Ec. (A.??) [eq:condition-number].
    clipped = clip_spectrum(eigenvalues, rtol=rtol, warn=False)
    lambda_max = float(clipped.max())
    lambda_min = float(clipped.min())
    positive = clipped[clipped > 0]
    cond = lambda_max / lambda_min if lambda_min > 0 else float("inf")
    # Numero de condicion efectivo (sobre el subespacio retenido):
    # Ec. (A.??) [eq:effective-cond].
    effective = lambda_max / float(positive.min()) if positive.size else float("inf")
    # kappa_2(K + eps I) = (lambda_max + eps) / (lambda_min + eps) <= 1 + lambda_max / eps:
    # Ec. (A.??) [eq:jitter-cond].
    if lambda_min + jitter > 0:
        cond_jitter = (lambda_max + jitter) / (lambda_min + jitter)
    else:
        cond_jitter = float("inf")
    return SpectralSummary(
        n=int(K.shape[0]),
        lambda_max=lambda_max,
        lambda_min=lambda_min,
        numerical_rank=int(positive.size),
        condition_number=float(cond),
        effective_condition_number=float(effective),
        jitter=float(jitter),
        condition_number_with_jitter=float(cond_jitter),
    )
