"""Composable positive-definite kernels with vectorised Gram matrices.

Design
------
* :class:`Kernel` is an abstract base class.  A concrete kernel only has to
  implement ``_gram(X, Y)`` (the cross Gram matrix, fully vectorised) and
  ``params()``; everything else (input validation, symmetrisation, ``+``,
  ``*``, scaling by a non-negative constant) is inherited.
* Kernels are stateless: they hold hyper-parameters but never see the data
  until ``gram`` is called.  That makes them trivially reusable between the
  SVM, Kernel PCA and the tests.
* :func:`register_kernel` / :func:`build_kernel` map the ``name`` used in the
  YAML configuration to a class, so adding a kernel requires no change in the
  rest of the code base (see the README, "Como agregar un kernel nuevo").

Closure under sums and products is not a programming convenience but a
theorem (Schur product theorem); see the appendix of ``docs/report.tex``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from numbers import Real

import numpy as np

from rkhs_credit.numerics import as_2d_float, symmetrize


class Kernel(ABC):
    """Abstract positive-definite kernel ``k: X x X -> R``."""

    # ------------------------------------------------------------------ API
    def gram(self, X, Y=None) -> np.ndarray:
        """Gram matrix ``K[i, j] = k(X[i], Y[j])``.

        Parameters
        ----------
        X : array-like of shape (n, d)
        Y : array-like of shape (m, d), optional
            If omitted, ``Y = X`` and the result is symmetrised to remove
            round-off asymmetries.

        Returns
        -------
        K : ndarray of shape (n, m)
        """
        X = as_2d_float(X, "X")
        if Y is None:
            # K(X, X) es simetrica semidefinida positiva por definicion de kernel
            # definido positivo, Ec. (A.1) [eq:pd-kernel]; en punto flotante se
            # fuerza la simetria con Ec. (A.78) [eq:symmetrization].
            return symmetrize(self._gram(X, X))
        Y = as_2d_float(Y, "Y")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"X and Y must have the same number of features; got {X.shape[1]} and {Y.shape[1]}."
            )
        return self._gram(X, Y)

    def __call__(self, X, Y=None) -> np.ndarray:
        """Alias of :meth:`gram`, so that ``k(X, Y)`` reads like the maths."""
        return self.gram(X, Y)

    def diag(self, X) -> np.ndarray:
        """Vector ``k(x_i, x_i)``.  Subclasses override it with an O(n) version."""
        return np.diag(self.gram(X)).copy()

    # ------------------------------------------------- to be implemented
    @abstractmethod
    def _gram(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Cross Gram matrix for validated 2-D float arrays (no Python loops)."""

    @abstractmethod
    def params(self) -> dict:
        """Hyper-parameters as a plain dict (used by ``repr`` and for logging)."""

    # --------------------------------------------------------- composition
    def __add__(self, other):
        if isinstance(other, Kernel):
            return SumKernel(self, other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Kernel):
            return ProductKernel(self, other)
        if isinstance(other, Real) and not isinstance(other, bool):
            return ScaledKernel(self, float(other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Real) and not isinstance(other, bool):
            return ScaledKernel(self, float(other))
        return NotImplemented

    def __repr__(self) -> str:
        args = ", ".join(f"{key}={value!r}" for key, value in self.params().items())
        return f"{type(self).__name__}({args})"


# ---------------------------------------------------------------- registry
_KERNEL_REGISTRY: dict[str, type[Kernel]] = {}


def register_kernel(name: str) -> Callable[[type[Kernel]], type[Kernel]]:
    """Class decorator that makes a kernel available to :func:`build_kernel`."""

    def decorator(cls: type[Kernel]) -> type[Kernel]:
        if name in _KERNEL_REGISTRY:
            raise ValueError(f"A kernel named {name!r} is already registered.")
        _KERNEL_REGISTRY[name] = cls
        return cls

    return decorator


def available_kernels() -> list[str]:
    """Names accepted by :func:`build_kernel` (besides the composite ones)."""
    return sorted(_KERNEL_REGISTRY)


def build_kernel(spec: Mapping) -> Kernel:
    """Build a kernel from a (possibly nested) configuration mapping.

    Examples
    --------
    >>> build_kernel({"name": "rbf", "gamma": 0.1})
    RBFKernel(gamma=0.1)
    >>> build_kernel({"name": "sum", "terms": [{"name": "linear"},
    ...                                        {"name": "rbf", "gamma": 0.1}]})
    SumKernel(left=LinearKernel(c=0.0), right=RBFKernel(gamma=0.1))

    Composite specs: ``{"name": "sum", "terms": [...]}``,
    ``{"name": "product", "factors": [...]}`` and
    ``{"name": "scaled", "scale": a, "kernel": {...}}``.
    """
    spec = dict(spec)
    try:
        name = spec.pop("name")
    except KeyError as exc:
        raise ValueError(f"Kernel spec {spec!r} has no 'name' entry.") from exc

    if name in ("sum", "product"):
        key = "terms" if name == "sum" else "factors"
        parts = [build_kernel(part) for part in spec.get(key, [])]
        if len(parts) < 2:
            raise ValueError(f"A {name!r} kernel needs at least two entries under {key!r}.")
        kernel = parts[0]
        for part in parts[1:]:
            kernel = kernel + part if name == "sum" else kernel * part
        return kernel
    if name == "scaled":
        return ScaledKernel(build_kernel(spec["kernel"]), float(spec["scale"]))
    if name not in _KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel {name!r}. Available: {available_kernels()}.")
    return _KERNEL_REGISTRY[name](**spec)


# ----------------------------------------------------------- base kernels
def squared_euclidean_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Matrix of squared distances ``D[i, j] = ||X[i] - Y[j]||^2`` (vectorised)."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x†y, Ec. (A.82) [eq:sqdist-expansion].
    # La resta sufre cancelacion cuando x ~ y, asi que los valores negativos
    # (del orden del error de redondeo) se recortan a cero.
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    y_sq = np.einsum("ij,ij->i", Y, Y)[None, :]
    D = x_sq + y_sq - 2.0 * (X @ Y.T)
    np.maximum(D, 0.0, out=D)
    if X is Y:
        # La distancia de un punto a si mismo es exactamente cero.
        np.fill_diagonal(D, 0.0)
    return D


@register_kernel("linear")
class LinearKernel(Kernel):
    """Linear kernel ``k(x, y) = x^T y + c`` with ``c >= 0``."""

    def __init__(self, c: float = 0.0):
        if c < 0:
            raise ValueError("The offset c must be non-negative for the kernel to be PSD.")
        self.c = float(c)

    def _gram(self, X, Y):
        # El producto interno euclideo es el kernel con mapa de caracteristicas
        # identidad, Ec. (A.2) [eq:feature-kernel]; sumar c >= 0 es sumar el kernel
        # constante (suma de kernels, Ec. (A.17) [eq:sum-kernel]).
        return X @ Y.T + self.c

    def diag(self, X):
        X = as_2d_float(X, "X")
        return np.einsum("ij,ij->i", X, X) + self.c

    def params(self):
        return {"c": self.c}


@register_kernel("rbf")
class RBFKernel(Kernel):
    """Gaussian (RBF) kernel ``k(x, y) = exp(-gamma * ||x - y||^2)``."""

    def __init__(self, gamma: float = 1.0):
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        self.gamma = float(gamma)

    def _gram(self, X, Y):
        # Es definido positivo por la factorizacion
        # exp(-g||x-y||^2) = exp(-g||x||^2) exp(-g||y||^2) exp(2g x†y),
        # Ec. (A.20) [eq:rbf-factorization].
        return np.exp(-self.gamma * squared_euclidean_distances(X, Y))

    def diag(self, X):
        X = as_2d_float(X, "X")
        return np.ones(X.shape[0])

    def params(self):
        return {"gamma": self.gamma}


@register_kernel("polynomial")
class PolynomialKernel(Kernel):
    """Polynomial kernel ``k(x, y) = (gamma * x^T y + coef0)^degree``."""

    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0):
        if int(degree) != degree or degree < 1:
            raise ValueError("degree must be a positive integer.")
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        if coef0 < 0:
            raise ValueError("coef0 must be non-negative for the kernel to be PSD.")
        self.degree = int(degree)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)

    def _gram(self, X, Y):
        # Potencia entera de un kernel con coeficientes no negativos: es kernel
        # por el teorema del producto de Schur, Ec. (A.19) [eq:poly-expansion].
        return (self.gamma * (X @ Y.T) + self.coef0) ** self.degree

    def diag(self, X):
        X = as_2d_float(X, "X")
        return (self.gamma * np.einsum("ij,ij->i", X, X) + self.coef0) ** self.degree

    def params(self):
        return {"degree": self.degree, "gamma": self.gamma, "coef0": self.coef0}


# ------------------------------------------------------ composite kernels
class SumKernel(Kernel):
    """Sum of two kernels, ``(k1 + k2)(x, y) = k1(x, y) + k2(x, y)``."""

    def __init__(self, left: Kernel, right: Kernel):
        self.left = left
        self.right = right

    def _gram(self, X, Y):
        # c†(K1 + K2)c = c†K1 c + c†K2 c >= 0, Ec. (A.17) [eq:sum-kernel].
        return self.left._gram(X, Y) + self.right._gram(X, Y)

    def diag(self, X):
        return self.left.diag(X) + self.right.diag(X)

    def params(self):
        return {"left": self.left, "right": self.right}


class ProductKernel(Kernel):
    """Product of two kernels, ``(k1 * k2)(x, y) = k1(x, y) * k2(x, y)``."""

    def __init__(self, left: Kernel, right: Kernel):
        self.left = left
        self.right = right

    def _gram(self, X, Y):
        # La matriz de Gram es el producto de Hadamard K1 o K2, que es semidefinido
        # positivo por el teorema del producto de Schur, Ec. (A.18) [eq:schur].
        return self.left._gram(X, Y) * self.right._gram(X, Y)

    def diag(self, X):
        return self.left.diag(X) * self.right.diag(X)

    def params(self):
        return {"left": self.left, "right": self.right}


class ScaledKernel(Kernel):
    """Kernel multiplied by a non-negative constant, ``a * k(x, y)``."""

    def __init__(self, kernel: Kernel, scale: float):
        if scale < 0:
            raise ValueError("The scale must be non-negative for the kernel to stay PSD.")
        self.kernel = kernel
        self.scale = float(scale)

    def _gram(self, X, Y):
        # c†(aK)c = a c†Kc >= 0 si a >= 0, Ec. (A.17) [eq:sum-kernel].
        return self.scale * self.kernel._gram(X, Y)

    def diag(self, X):
        return self.scale * self.kernel.diag(X)

    def params(self):
        return {"kernel": self.kernel, "scale": self.scale}
