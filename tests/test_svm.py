"""Kernel SVM / SMO: agreement with scikit-learn and optimality conditions."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import optimize
from sklearn.svm import SVC

from rkhs_credit.kernels import LinearKernel, PolynomialKernel, RBFKernel
from rkhs_credit.svm import KernelSVM, smo

TOL = 1e-3

# (kernel propio, argumentos equivalentes de sklearn.svm.SVC)
KERNEL_PAIRS = {
    "rbf": (RBFKernel(gamma=0.2), {"kernel": "rbf", "gamma": 0.2}),
    "linear": (LinearKernel(), {"kernel": "linear"}),
    "polynomial": (
        PolynomialKernel(degree=3, gamma=0.1, coef0=1.0),
        {"kernel": "poly", "degree": 3, "gamma": 0.1, "coef0": 1.0},
    ),
}
CLASS_WEIGHTS = {"unweighted": None, "cost-5-to-1": {0: 1.0, 1: 5.0}, "balanced": "balanced"}


@pytest.mark.parametrize("weights", list(CLASS_WEIGHTS))
@pytest.mark.parametrize("name", list(KERNEL_PAIRS))
def test_matches_sklearn_svc(name, weights, two_class_data):
    X, y, X_new = two_class_data
    kernel, sklearn_kwargs = KERNEL_PAIRS[name]
    class_weight = CLASS_WEIGHTS[weights]

    own = KernelSVM(kernel, C=1.0, class_weight=class_weight, tol=TOL).fit(X, y)
    reference = SVC(C=1.0, class_weight=class_weight, tol=TOL, **sklearn_kwargs).fit(X, y)

    own_scores = own.decision_function(X_new)
    reference_scores = reference.decision_function(X_new)
    agreement = np.mean(own.predict(X_new) == reference.predict(X_new))
    correlation = np.corrcoef(own_scores, reference_scores)[0, 1]

    assert own.converged_
    assert agreement >= 0.98
    assert correlation >= 0.999
    np.testing.assert_allclose(own_scores, reference_scores, atol=0.02)
    assert own.intercept_ == pytest.approx(reference.intercept_[0], abs=0.02)
    assert abs(own.support_.size - reference.support_.size) <= 0.05 * reference.support_.size


def test_matches_sklearn_with_a_composite_kernel(two_class_data):
    """Same Gram matrix on both sides: sklearn receives the kernel as a callable."""
    X, y, X_new = two_class_data
    kernel = 0.5 * LinearKernel() + RBFKernel(0.3) * PolynomialKernel(2, gamma=0.2)
    own = KernelSVM(kernel, C=2.0, class_weight={0: 1.0, 1: 5.0}, tol=TOL).fit(X, y)
    reference = SVC(C=2.0, kernel=kernel.gram, class_weight={0: 1.0, 1: 5.0}, tol=TOL).fit(X, y)
    assert np.mean(own.predict(X_new) == reference.predict(X_new)) >= 0.98
    correlation = np.corrcoef(own.decision_function(X_new), reference.decision_function(X_new))
    assert correlation[0, 1] >= 0.999


def test_dual_objective_matches_sklearn(two_class_data):
    X, y, _ = two_class_data
    kernel = RBFKernel(0.2)
    own = KernelSVM(kernel, C=1.0, tol=1e-4).fit(X, y)
    reference = SVC(C=1.0, kernel="rbf", gamma=0.2, tol=1e-4).fit(X, y)
    coefficients = reference.dual_coef_.ravel()  # alpha_i y_i de los vectores de soporte
    gram = kernel.gram(X[reference.support_])
    reference_objective = np.abs(coefficients).sum() - 0.5 * coefficients @ gram @ coefficients
    assert own.dual_objective_ == pytest.approx(reference_objective, rel=1e-4)


def test_solution_satisfies_the_kkt_conditions(two_class_data):
    X, y, _ = two_class_data
    C, weights = 1.5, {0: 1.0, 1: 5.0}
    model = KernelSVM(RBFKernel(0.2), C=C, class_weight=weights, tol=TOL).fit(X, y)
    y_signed = np.where(y == 1, 1.0, -1.0)
    upper = C * np.where(y == 1, weights[1], weights[0])
    alpha = model.alpha_

    # Factibilidad dual: caja y restriccion de igualdad.
    assert np.all(alpha >= 0.0)
    assert np.all(alpha <= upper)
    assert abs(alpha @ y_signed) <= 1e-8 * upper.max()
    assert model.kkt_gap_ <= TOL

    # Holgura complementaria, en terminos del margen funcional y_i f(x_i).
    margins = y_signed * model.decision_function(X)
    at_zero = alpha == 0.0
    at_upper = alpha == upper
    free = ~at_zero & ~at_upper
    assert np.all(margins[at_zero] >= 1.0 - TOL)
    assert np.all(margins[at_upper] <= 1.0 + TOL)
    np.testing.assert_allclose(margins[free], 1.0, atol=TOL)
    assert free.sum() == model.n_free_support_
    np.testing.assert_array_equal(np.flatnonzero(~at_zero), model.support_)
    np.testing.assert_array_equal(np.flatnonzero(free), model.free_support_)


def test_smo_reaches_the_optimum_of_the_dual_qp(rng):
    """Independent check of :func:`smo` against a generic constrained optimiser."""
    X = rng.normal(size=(24, 2))
    y = np.where(X[:, 0] + 0.5 * rng.normal(size=24) > 0, 1.0, -1.0)
    K = RBFKernel(0.5).gram(X)
    upper = np.where(y > 0, 2.0, 0.5)
    Q = (y[:, None] * y[None, :]) * K

    def negative_dual(alpha):
        return 0.5 * alpha @ Q @ alpha - alpha.sum()

    reference = optimize.minimize(
        negative_dual,
        x0=np.zeros(24),
        jac=lambda alpha: Q @ alpha - 1.0,
        bounds=list(zip(np.zeros(24), upper, strict=True)),
        constraints=[{"type": "eq", "fun": lambda alpha: alpha @ y, "jac": lambda alpha: y}],
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 500},
    )
    result = smo(K, y, upper, tol=1e-6, max_iter=100_000)
    assert result.converged
    assert abs(result.alpha @ y) <= 1e-9
    assert -negative_dual(result.alpha) == pytest.approx(-reference.fun, rel=1e-5)
    np.testing.assert_allclose(result.alpha, reference.x, atol=2e-3)
    # El cache de errores es consistente con su definicion E = K (alpha * y) - y.
    np.testing.assert_allclose(result.errors, K @ (result.alpha * y) - y, atol=1e-9)


def test_class_weights_shift_the_decision_towards_the_expensive_class(two_class_data):
    X, y, X_new = two_class_data
    unweighted = KernelSVM(RBFKernel(0.2), C=1.0).fit(X, y)
    weighted = KernelSVM(RBFKernel(0.2), C=1.0, class_weight={0: 1.0, 1: 5.0}).fit(X, y)
    assert weighted.predict(X_new).mean() > unweighted.predict(X_new).mean()
    recall = lambda model: model.predict(X)[y == 1].mean()  # noqa: E731
    assert recall(weighted) >= recall(unweighted)


def test_separable_problem_is_solved_exactly_with_few_support_vectors(rng):
    X = np.vstack([rng.normal(-3.0, 0.5, size=(40, 2)), rng.normal(3.0, 0.5, size=(40, 2))])
    y = np.repeat([0, 1], 40)
    model = KernelSVM(LinearKernel(), C=10.0).fit(X, y)
    assert np.array_equal(model.predict(X), y)
    assert model.support_.size <= 6
    margins = np.where(y == 1, 1.0, -1.0) * model.decision_function(X)
    assert margins.min() >= 1.0 - 1e-2  # margen duro: todos los puntos fuera del margen


def test_arbitrary_labels_are_preserved(two_class_data):
    X, y, X_new = two_class_data
    labels = np.where(y == 1, "bad", "good")
    model = KernelSVM(RBFKernel(0.2), class_weight={"bad": 5.0, "good": 1.0}).fit(X, labels)
    assert list(model.classes_) == ["bad", "good"]
    assert set(model.predict(X_new)) <= {"bad", "good"}
    # classes_[1] ("good") es la clase positiva: f > 0  <=>  "good".
    scores = model.decision_function(X_new)
    assert np.array_equal(model.predict(X_new) == "good", scores > 0)


def test_duplicated_points_do_not_break_the_solver(rng):
    """Identical points give eta = 0; the jitter / eta floor must handle it."""
    X = rng.normal(size=(30, 2))
    X = np.vstack([X, X])
    y = np.concatenate([np.ones(30, dtype=int), np.zeros(30, dtype=int)])  # etiquetas opuestas
    for jitter in (0.0, 1e-8):
        model = KernelSVM(RBFKernel(1.0), C=1.0, jitter=jitter).fit(X, y)
        assert model.converged_
        assert np.all(np.isfinite(model.alpha_))


def test_max_iter_emits_a_warning(two_class_data):
    X, y, _ = two_class_data
    with pytest.warns(RuntimeWarning, match="max_iter"):
        model = KernelSVM(RBFKernel(0.2), max_iter=5).fit(X, y)
    assert not model.converged_
    assert model.n_iter_ == 5


def test_no_warning_when_converged(two_class_data):
    X, y, _ = two_class_data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KernelSVM(RBFKernel(0.2)).fit(X, y)


def test_input_errors(two_class_data):
    X, y, _ = two_class_data
    with pytest.raises(ValueError, match="binary"):
        KernelSVM(RBFKernel(1.0)).fit(X, np.arange(X.shape[0]) % 3)
    with pytest.raises(ValueError, match="inconsistent"):
        KernelSVM(RBFKernel(1.0)).fit(X, y[:-1])
    with pytest.raises(ValueError, match="no entry"):
        KernelSVM(RBFKernel(1.0), class_weight={1: 5.0}).fit(X, y)
    with pytest.raises(ValueError):
        KernelSVM(RBFKernel(1.0), C=0.0)
    with pytest.raises(RuntimeError):
        KernelSVM(RBFKernel(1.0)).decision_function(X)
    with pytest.raises(ValueError, match="-1 and \\+1"):
        smo(np.eye(3), np.array([0.0, 1.0, 1.0]), 1.0)
