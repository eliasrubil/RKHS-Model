"""Kernel PCA: agreement with scikit-learn and the identities of the appendix."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import KernelPCA as SklearnKernelPCA

from rkhs_credit.kernels import LinearKernel, PolynomialKernel, RBFKernel
from rkhs_credit.kpca import KernelPCA
from rkhs_credit.numerics import center_gram

# (kernel propio, argumentos equivalentes de sklearn)
KERNEL_PAIRS = {
    "rbf": (RBFKernel(gamma=0.15), {"kernel": "rbf", "gamma": 0.15}),
    "polynomial": (
        PolynomialKernel(degree=3, gamma=0.1, coef0=1.0),
        {"kernel": "poly", "degree": 3, "gamma": 0.1, "coef0": 1.0},
    ),
    "linear": (LinearKernel(), {"kernel": "linear"}),
}


@pytest.fixture
def data(rng):
    return rng.normal(size=(60, 5)), rng.normal(size=(25, 5))


def assert_equal_up_to_sign(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-8):
    """Columns may differ by a global sign: eigenvectors are defined up to +-1."""
    assert actual.shape == expected.shape
    signs = np.sign(np.sum(actual * expected, axis=0))
    np.testing.assert_allclose(actual * signs, expected, atol=atol)


@pytest.mark.parametrize("name", list(KERNEL_PAIRS))
def test_matches_sklearn_up_to_sign(name, data):
    X, X_new = data
    kernel, sklearn_kwargs = KERNEL_PAIRS[name]
    n_components = 4
    own = KernelPCA(kernel, n_components=n_components)
    reference = SklearnKernelPCA(n_components=n_components, eigen_solver="dense", **sklearn_kwargs)

    own_train = own.fit_transform(X)
    reference_train = reference.fit_transform(X)
    scale = np.abs(reference_train).max()
    assert_equal_up_to_sign(own_train, reference_train, atol=1e-8 * scale)
    assert_equal_up_to_sign(own.transform(X_new), reference.transform(X_new), atol=1e-8 * scale)
    np.testing.assert_allclose(own.eigenvalues_, reference.eigenvalues_, rtol=1e-8)


def test_centered_gram_has_zero_mean_rows_and_columns(data):
    X, _ = data
    K_centered = center_gram(PolynomialKernel(2).gram(X))
    scale = np.abs(K_centered).max()
    np.testing.assert_allclose(K_centered.mean(axis=0), 0.0, atol=1e-12 * scale)
    np.testing.assert_allclose(K_centered.mean(axis=1), 0.0, atol=1e-12 * scale)


def test_principal_directions_are_orthonormal_in_the_rkhs(data):
    """<v_k, v_l>_H = a_k^T K~ a_l must be the identity matrix."""
    X, _ = data
    model = KernelPCA(RBFKernel(0.15), n_components=6).fit(X)
    K_centered = center_gram(RBFKernel(0.15).gram(X))
    gram_of_directions = model.dual_coef_.T @ K_centered @ model.dual_coef_
    np.testing.assert_allclose(gram_of_directions, np.eye(6), atol=1e-9)


def test_transform_of_training_points_equals_fit_transform(data):
    X, _ = data
    model = KernelPCA(RBFKernel(0.15), n_components=5)
    scores = model.fit_transform(X)
    np.testing.assert_allclose(model.transform(X), scores, atol=1e-10)


def test_scores_are_centered_with_variance_lambda_over_n(data):
    X, _ = data
    model = KernelPCA(RBFKernel(0.15), n_components=5)
    scores = model.fit_transform(X)
    np.testing.assert_allclose(scores.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        (scores**2).mean(axis=0), model.eigenvalues_ / X.shape[0], rtol=1e-10
    )
    # Componentes distintas no estan correlacionadas.
    covariance = scores.T @ scores / X.shape[0]
    np.testing.assert_allclose(covariance - np.diag(np.diag(covariance)), 0.0, atol=1e-12)


def test_linear_kernel_recovers_classical_pca(data):
    X, X_new = data
    model = KernelPCA(LinearKernel(), n_components=3)
    scores = model.fit_transform(X)

    X_centered = X - X.mean(axis=0)
    _, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
    assert_equal_up_to_sign(scores, X_centered @ Vt[:3].T, atol=1e-9)
    assert_equal_up_to_sign(model.transform(X_new), (X_new - X.mean(axis=0)) @ Vt[:3].T, atol=1e-9)
    # Autovalores de K~ = cuadrados de los valores singulares de X centrada.
    np.testing.assert_allclose(model.eigenvalues_, singular_values[:3] ** 2, rtol=1e-10)


def test_all_components_are_kept_up_to_the_numerical_rank(data):
    X, _ = data  # d = 5
    linear = KernelPCA(LinearKernel()).fit(X)
    assert linear.eigenvalues_.size == 5  # rango de K~ = rango de X centrada
    assert linear.explained_variance_ratio_.sum() == pytest.approx(1.0)
    assert np.all(linear.spectrum_ >= 0.0)
    assert np.all(np.diff(linear.spectrum_) <= 0.0)  # orden decreciente

    rbf = KernelPCA(RBFKernel(0.15)).fit(X)
    assert rbf.eigenvalues_.size <= X.shape[0] - 1  # centrar quita al menos una dimension
    assert rbf.condition_number_ == pytest.approx(rbf.eigenvalues_[0] / rbf.eigenvalues_[-1])


def test_explained_variance_ratio_is_relative_to_the_trace(data):
    X, _ = data
    model = KernelPCA(RBFKernel(0.15), n_components=4).fit(X)
    trace = np.trace(center_gram(RBFKernel(0.15).gram(X)))
    np.testing.assert_allclose(model.explained_variance_ratio_, model.eigenvalues_ / trace)
    assert model.explained_variance_ratio_.sum() < 1.0


def test_eigenvector_signs_are_deterministic(data):
    X, _ = data
    first = KernelPCA(RBFKernel(0.15), n_components=4).fit_transform(X)
    second = KernelPCA(RBFKernel(0.15), n_components=4).fit_transform(X.copy())
    np.testing.assert_array_equal(first, second)


def test_errors():
    with pytest.raises(ValueError):
        KernelPCA(RBFKernel(1.0), n_components=0)
    with pytest.raises(RuntimeError):
        KernelPCA(RBFKernel(1.0)).transform(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="no positive eigenvalue"):
        KernelPCA(LinearKernel()).fit(np.ones((5, 3)))  # todos los puntos iguales
