"""Kernels: Gram matrices are symmetric PSD, vectorised code matches the definition."""

from __future__ import annotations

import numpy as np
import pytest

from rkhs_credit import kernels
from rkhs_credit.kernels import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    ProductKernel,
    RBFKernel,
    ScaledKernel,
    SumKernel,
    build_kernel,
    register_kernel,
)

PSD_RTOL = 1e-10  # autovalores >= -PSD_RTOL * lambda_max


def scalar_linear(x, y, c=0.0):
    return float(x @ y + c)


def scalar_rbf(x, y, gamma):
    return float(np.exp(-gamma * np.sum((x - y) ** 2)))


def scalar_poly(x, y, degree, gamma, coef0):
    return float((gamma * (x @ y) + coef0) ** degree)


# (kernel, funcion escalar equivalente evaluada de a un par de puntos)
KERNEL_CASES = {
    "linear": (LinearKernel(), lambda x, y: scalar_linear(x, y)),
    "linear+c": (LinearKernel(c=1.5), lambda x, y: scalar_linear(x, y, 1.5)),
    "rbf": (RBFKernel(gamma=0.3), lambda x, y: scalar_rbf(x, y, 0.3)),
    "poly2": (PolynomialKernel(degree=2), lambda x, y: scalar_poly(x, y, 2, 1.0, 1.0)),
    "poly3": (
        PolynomialKernel(degree=3, gamma=0.2, coef0=2.0),
        lambda x, y: scalar_poly(x, y, 3, 0.2, 2.0),
    ),
    "sum": (
        LinearKernel() + RBFKernel(0.3),
        lambda x, y: scalar_linear(x, y) + scalar_rbf(x, y, 0.3),
    ),
    "product": (
        RBFKernel(0.3) * PolynomialKernel(2),
        lambda x, y: scalar_rbf(x, y, 0.3) * scalar_poly(x, y, 2, 1.0, 1.0),
    ),
    "scaled": (2.5 * RBFKernel(0.3), lambda x, y: 2.5 * scalar_rbf(x, y, 0.3)),
    "nested": (
        (LinearKernel() + RBFKernel(1.0)) * PolynomialKernel(2) + 0.5 * RBFKernel(0.1),
        lambda x, y: (
            (scalar_linear(x, y) + scalar_rbf(x, y, 1.0)) * scalar_poly(x, y, 2, 1.0, 1.0)
            + 0.5 * scalar_rbf(x, y, 0.1)
        ),
    ),
}


@pytest.fixture(params=list(KERNEL_CASES), ids=list(KERNEL_CASES))
def kernel_case(request):
    return KERNEL_CASES[request.param]


def test_gram_is_symmetric_and_psd(kernel_case, rng):
    kernel, _ = kernel_case
    X = rng.normal(size=(40, 6))
    K = kernel.gram(X)
    assert K.shape == (40, 40)
    assert np.array_equal(K, K.T)
    eigenvalues = np.linalg.eigvalsh(K)
    assert eigenvalues.min() >= -PSD_RTOL * eigenvalues.max()


def test_gram_is_psd_with_duplicated_points(kernel_case, rng):
    """Repeated rows make the Gram matrix singular; it must stay PSD up to round-off."""
    kernel, _ = kernel_case
    X = rng.normal(size=(15, 4))
    X = np.vstack([X, X[:5]])
    eigenvalues = np.linalg.eigvalsh(kernel.gram(X))
    assert eigenvalues.min() >= -PSD_RTOL * eigenvalues.max()


def test_gram_matches_pairwise_definition(kernel_case, rng):
    kernel, scalar = kernel_case
    X = rng.normal(size=(7, 3))
    Y = rng.normal(size=(5, 3))
    expected = np.array([[scalar(x, y) for y in Y] for x in X])
    np.testing.assert_allclose(kernel.gram(X, Y), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(kernel(X, Y), expected, rtol=1e-10, atol=1e-12)


def test_cross_gram_is_transpose_consistent(kernel_case, rng):
    kernel, _ = kernel_case
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(6, 3))
    np.testing.assert_allclose(kernel.gram(X, Y), kernel.gram(Y, X).T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(kernel.gram(X), kernel.gram(X, X.copy()), rtol=1e-10, atol=1e-10)


def test_diag_matches_gram_diagonal(kernel_case, rng):
    kernel, _ = kernel_case
    X = rng.normal(size=(12, 4))
    np.testing.assert_allclose(kernel.diag(X), np.diag(kernel.gram(X)), rtol=1e-10, atol=1e-12)


def test_sum_product_and_scaling_act_entrywise(rng):
    X = rng.normal(size=(10, 3))
    Y = rng.normal(size=(4, 3))
    k1, k2 = RBFKernel(0.7), PolynomialKernel(2, gamma=0.5)
    np.testing.assert_allclose((k1 + k2).gram(X, Y), k1.gram(X, Y) + k2.gram(X, Y))
    np.testing.assert_allclose((k1 * k2).gram(X, Y), k1.gram(X, Y) * k2.gram(X, Y))
    np.testing.assert_allclose((3.0 * k1).gram(X, Y), 3.0 * k1.gram(X, Y))
    np.testing.assert_allclose((k1 * 3).gram(X, Y), 3.0 * k1.gram(X, Y))


def test_operators_return_composite_kernels():
    k1, k2 = LinearKernel(), RBFKernel(1.0)
    assert isinstance(k1 + k2, SumKernel)
    assert isinstance(k1 * k2, ProductKernel)
    assert isinstance(2 * k1, ScaledKernel)
    assert isinstance(k1 * 2.0, ScaledKernel)
    with pytest.raises(TypeError):
        _ = k1 + 1.0
    with pytest.raises(TypeError):
        _ = "a" * k1


def test_schur_product_of_psd_matrices_is_psd(rng):
    """Numerical illustration of the Schur product theorem used by ProductKernel."""
    A = rng.normal(size=(12, 4))
    B = rng.normal(size=(12, 3))
    hadamard = (A @ A.T) * (B @ B.T)
    eigenvalues = np.linalg.eigvalsh(hadamard)
    assert eigenvalues.min() >= -PSD_RTOL * eigenvalues.max()


def test_rbf_kernel_properties(rng):
    X = rng.normal(size=(9, 2))
    K = RBFKernel(gamma=2.0).gram(X)
    assert np.all(np.diag(K) == 1.0)
    assert np.all((K > 0) & (K <= 1.0))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RBFKernel(gamma=0.0),
        lambda: RBFKernel(gamma=-1.0),
        lambda: LinearKernel(c=-0.1),
        lambda: PolynomialKernel(degree=0),
        lambda: PolynomialKernel(degree=2.5),
        lambda: PolynomialKernel(degree=2, coef0=-1.0),
        lambda: ScaledKernel(RBFKernel(1.0), -2.0),
    ],
)
def test_invalid_hyperparameters_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_input_validation(rng):
    kernel = RBFKernel(1.0)
    with pytest.raises(ValueError):
        kernel.gram(rng.normal(size=5))  # 1-D
    with pytest.raises(ValueError):
        kernel.gram(rng.normal(size=(4, 3)), rng.normal(size=(4, 2)))  # distinta dimension


def test_build_kernel_from_config_specs(rng):
    X = rng.normal(size=(6, 3))
    rbf = build_kernel({"name": "rbf", "gamma": 0.25})
    assert isinstance(rbf, RBFKernel) and rbf.gamma == 0.25

    spec = {
        "name": "sum",
        "terms": [
            {"name": "linear"},
            {
                "name": "product",
                "factors": [
                    {"name": "rbf", "gamma": 0.5},
                    {"name": "polynomial", "degree": 2},
                ],
            },
            {"name": "scaled", "scale": 0.1, "kernel": {"name": "rbf", "gamma": 2.0}},
        ],
    }
    expected = (
        LinearKernel().gram(X)
        + RBFKernel(0.5).gram(X) * PolynomialKernel(2).gram(X)
        + 0.1 * RBFKernel(2.0).gram(X)
    )
    np.testing.assert_allclose(build_kernel(spec).gram(X), expected)


def test_build_kernel_errors():
    with pytest.raises(ValueError, match="Unknown kernel"):
        build_kernel({"name": "does-not-exist"})
    with pytest.raises(ValueError, match="no 'name'"):
        build_kernel({"gamma": 1.0})
    with pytest.raises(ValueError, match="at least two"):
        build_kernel({"name": "sum", "terms": [{"name": "linear"}]})


def test_new_kernel_can_be_registered_and_composed(rng):
    """The extension path described in the README: subclass + ``register_kernel``."""

    @register_kernel("test-laplacian")
    class LaplacianKernel(Kernel):
        def __init__(self, gamma: float = 1.0):
            self.gamma = gamma

        def _gram(self, X, Y):
            distances = np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
            return np.exp(-self.gamma * distances)

        def params(self):
            return {"gamma": self.gamma}

    try:
        kernel = build_kernel({"name": "test-laplacian", "gamma": 0.5}) + RBFKernel(1.0)
        X = rng.normal(size=(20, 3))
        K = kernel.gram(X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.array_equal(K, K.T)
        assert eigenvalues.min() >= -PSD_RTOL * eigenvalues.max()
        np.testing.assert_allclose(kernel.diag(X), np.diag(K))  # diag() heredado
        with pytest.raises(ValueError, match="already registered"):
            register_kernel("test-laplacian")(LaplacianKernel)
    finally:
        kernels._KERNEL_REGISTRY.pop("test-laplacian", None)
