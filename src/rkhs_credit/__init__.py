"""Kernel methods (RKHS) for credit-risk classification.

The package implements, with NumPy/SciPy only, the three core pieces of the
project and the glue around them:

* :mod:`rkhs_credit.kernels`  -- composable positive-definite kernels.
* :mod:`rkhs_credit.svm`      -- soft-margin SVM trained in the dual with SMO.
* :mod:`rkhs_credit.kpca`     -- Kernel PCA.
* :mod:`rkhs_credit.numerics` -- small numerical-stability helpers.
* :mod:`rkhs_credit.data`, :mod:`rkhs_credit.pipeline`,
  :mod:`rkhs_credit.tuning`, :mod:`rkhs_credit.evaluation`,
  :mod:`rkhs_credit.plots` -- data handling, model assembly and reporting.

Mathematical comments in the source are written in Spanish and point to the
numbered equations of the appendix of ``docs/report.tex``.
"""

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
from rkhs_credit.kpca import KernelPCA
from rkhs_credit.svm import KernelSVM, smo

__version__ = "0.1.0"

__all__ = [
    "Kernel",
    "KernelPCA",
    "KernelSVM",
    "LinearKernel",
    "PolynomialKernel",
    "ProductKernel",
    "RBFKernel",
    "ScaledKernel",
    "SumKernel",
    "build_kernel",
    "register_kernel",
    "smo",
]
