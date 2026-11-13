"""Kernel PCA: principal component analysis in a reproducing kernel Hilbert space.

The implementation follows Scholkopf, Smola and Muller (1998) step by step:

1. Gram matrix ``K`` of the training sample.
2. Centering in feature space, ``K~ = H K H``.
3. Symmetric eigendecomposition of ``K~`` (``scipy.linalg.eigh``).
4. Normalisation of the eigenvectors so that every principal direction has
   unit norm *in the RKHS*.
5. Projection of training and new points (the test kernel matrix is centred
   with the training mean).

Linear PCA is the particular case ``kernel = LinearKernel()``, which is how
the figures of the report obtain their "PCA lineal" panel.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from rkhs_credit.kernels import Kernel
from rkhs_credit.numerics import (
    DEFAULT_EIG_RTOL,
    as_2d_float,
    center_gram,
    center_gram_test,
    clip_spectrum,
)


class KernelPCA:
    """Kernel principal component analysis.

    Parameters
    ----------
    kernel : Kernel
        Positive-definite kernel.
    n_components : int or None
        Number of components to keep.  ``None`` keeps every component whose
        eigenvalue is numerically positive.
    eig_rtol : float
        Eigenvalues below ``eig_rtol * lambda_max`` are treated as zero.

    Attributes
    ----------
    eigenvalues_ : ndarray of shape (q,)
        Eigenvalues of the centred Gram matrix for the retained components,
        in decreasing order.
    spectrum_ : ndarray of shape (n,)
        Full (clipped) spectrum of the centred Gram matrix, decreasing.
    eigenvectors_ : ndarray of shape (n, q)
        Orthonormal eigenvectors ``u_k`` of the centred Gram matrix.
    dual_coef_ : ndarray of shape (n, q)
        Expansion coefficients ``a_k = u_k / sqrt(lambda_k)`` of the unit-norm
        principal directions.
    explained_variance_ratio_ : ndarray of shape (q,)
        ``lambda_k / trace(K~)``.
    condition_number_ : float
        ``lambda_1 / lambda_q`` over the retained components.
    """

    def __init__(
        self,
        kernel: Kernel,
        n_components: int | None = None,
        eig_rtol: float = DEFAULT_EIG_RTOL,
    ):
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be a positive integer or None.")
        self.kernel = kernel
        self.n_components = n_components
        self.eig_rtol = float(eig_rtol)

    # ---------------------------------------------------------------- fit
    def fit(self, X) -> KernelPCA:
        X = as_2d_float(X, "X")
        K = self.kernel.gram(X)  # simetrica, Ec. (A.??) [eq:symmetrization]
        # Centrado en el espacio de caracteristicas: K~ = H K H,
        # Ec. (A.??) [eq:centering]. Filas y columnas de K~ suman cero,
        # Ec. (A.??) [eq:centered-zero-sums].
        K_centered = center_gram(K)

        # Problema de autovalores K~ a = lambda a, Ec. (A.??) [eq:kpca-gram-eigen],
        # es decir det(K~ - lambda I) = 0, Ec. (A.??) [eq:kpca-char-eq].
        # K~ es simetrica real: autovalores reales y autovectores ortonormales,
        # Ec. (A.??) [eq:spectral-decomposition]. eigh los devuelve en orden creciente.
        eigenvalues, eigenvectors = linalg.eigh(K_centered)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Autovalores negativos del orden del redondeo se tratan como cero,
        # Ec. (A.??) [eq:eig-threshold].
        spectrum = clip_spectrum(eigenvalues, rtol=self.eig_rtol)
        n_positive = int(np.count_nonzero(spectrum))
        if n_positive == 0:
            raise ValueError("The centred Gram matrix has no positive eigenvalue.")
        q = n_positive if self.n_components is None else min(self.n_components, n_positive)

        retained_values = spectrum[:q]
        retained_vectors = _fix_signs(eigenvectors[:, :q])

        self.X_fit_ = X
        self.K_fit_ = K
        self.spectrum_ = spectrum
        self.eigenvalues_ = retained_values
        self.eigenvectors_ = retained_vectors
        # Normalizacion: ||v_k||^2 = lambda_k a_k†a_k = 1  =>  a_k = u_k / sqrt(lambda_k),
        # Ec. (A.??) [eq:kpca-normalization].
        self.dual_coef_ = retained_vectors / np.sqrt(retained_values)
        # Fraccion de varianza explicada: lambda_k / tr(K~), Ec. (A.??) [eq:explained-variance].
        self.explained_variance_ratio_ = retained_values / spectrum.sum()
        # Numero de condicion efectivo de las componentes retenidas,
        # Ec. (A.??) [eq:effective-cond].
        self.condition_number_ = float(retained_values[0] / retained_values[-1])
        return self

    # ---------------------------------------------------------- transform
    def transform(self, X) -> np.ndarray:
        """Project points onto the retained principal directions."""
        self._check_is_fitted()
        X = as_2d_float(X, "X")
        K_test = self.kernel.gram(X, self.X_fit_)
        # La matriz de kernel de test se centra con la media de train,
        # Ec. (A.??) [eq:centering-test].
        K_test_centered = center_gram_test(K_test, self.K_fit_)
        # z_k(x) = sum_i a_{k,i} k~(x, x_i), Ec. (A.??) [eq:kpca-new-point];
        # en forma matricial Z = K~_test A, Ec. (A.??) [eq:kpca-test-projection].
        return K_test_centered @ self.dual_coef_

    def fit_transform(self, X) -> np.ndarray:
        """Fit and return the projections of the training points."""
        self.fit(X)
        # Para los puntos de train no hace falta recalcular el kernel:
        # z_k(x_i) = sqrt(lambda_k) u_{k,i}, Ec. (A.??) [eq:kpca-train-projection].
        return self.eigenvectors_ * np.sqrt(self.eigenvalues_)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("This KernelPCA instance is not fitted yet; call fit first.")


def _fix_signs(eigenvectors: np.ndarray) -> np.ndarray:
    """Make the sign of each eigenvector deterministic.

    An eigenvector is only defined up to sign.  The entry of largest absolute
    value is made positive (the same convention scikit-learn uses), so figures
    and results are reproducible across LAPACK builds.
    """
    pivot_rows = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[pivot_rows, np.arange(eigenvectors.shape[1])])
    signs[signs == 0] = 1.0
    return eigenvectors * signs
