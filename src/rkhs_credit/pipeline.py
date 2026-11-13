"""Model assembly: preprocessing -> (optional) Kernel PCA -> classifier.

Every model in the project (the kernel SVM, its Kernel PCA variant and the
baselines) exposes the same three methods on a raw ``pandas`` frame::

    model.fit(X_frame, y)
    model.decision_function(X_frame)   # higher = riskier
    model.predict(X_frame)             # 1 = bad, 0 = good

so that cross-validation, evaluation and plotting code never needs to know
which model it is handling.

:func:`build_pipeline` is the only place where the YAML configuration is
turned into objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from rkhs_credit.data import TabularPreprocessor
from rkhs_credit.evaluation import CostMatrix
from rkhs_credit.kernels import build_kernel
from rkhs_credit.kpca import KernelPCA
from rkhs_credit.svm import KernelSVM


class CreditPipeline:
    """Preprocessing, optional Kernel PCA and a kernel SVM chained together.

    Parameters
    ----------
    svm : KernelSVM
    kpca : KernelPCA or None
        When given, the SVM is trained on the Kernel PCA scores instead of the
        preprocessed attributes.
    exclude : sequence of str
        Raw columns removed before preprocessing (sensitive attributes).

    Attributes
    ----------
    preprocessor_ : TabularPreprocessor
        Fitted on the data passed to :meth:`fit` only.
    """

    def __init__(self, svm: KernelSVM, kpca: KernelPCA | None = None, exclude: Sequence[str] = ()):
        self.svm = svm
        self.kpca = kpca
        self.exclude = tuple(exclude)

    def fit(self, X: pd.DataFrame, y) -> CreditPipeline:
        # Todo lo que se "aprende" (medias, desvios, categorias, autovectores de
        # Kernel PCA, multiplicadores de la SVM) sale unicamente de (X, y): si se
        # llama con el conjunto de train, no hay fuga de informacion del test.
        self.preprocessor_ = TabularPreprocessor(exclude=self.exclude)
        features = self.preprocessor_.fit_transform(X)
        if self.kpca is not None:
            features = self.kpca.fit_transform(features)
        self.svm.fit(features, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Features seen by the SVM (preprocessed and, if enabled, projected)."""
        features = self.preprocessor_.transform(X)
        if self.kpca is not None:
            features = self.kpca.transform(features)
        return features

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.svm.decision_function(self.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.svm.predict(self.transform(X))


def resolve_class_weight(setting, cost: CostMatrix):
    """Translate the ``class_weight`` entry of the configuration.

    ``"cost_matrix"`` -> weights taken from the cost matrix;
    ``"balanced"`` / ``None`` / an explicit mapping are passed through.
    """
    if setting == "cost_matrix":
        return cost.class_weight()
    if setting in (None, "none"):
        return None
    if isinstance(setting, Mapping):
        return {int(label): float(weight) for label, weight in setting.items()}
    return setting


def excluded_columns(config: Mapping) -> tuple[str, ...]:
    """Raw columns to drop according to ``data.exclude_sensitive``."""
    data_cfg = config["data"]
    if data_cfg.get("exclude_sensitive", False):
        return tuple(data_cfg["sensitive_features"])
    return ()


def build_pipeline(config: Mapping) -> CreditPipeline:
    """Create an unfitted :class:`CreditPipeline` from a configuration dict."""
    cost = CostMatrix.from_config(config)
    svm_cfg = config["svm"]
    svm = KernelSVM(
        kernel=build_kernel(svm_cfg["kernel"]),
        C=svm_cfg["C"],
        class_weight=resolve_class_weight(svm_cfg.get("class_weight"), cost),
        tol=svm_cfg["tol"],
        max_iter=svm_cfg["max_iter"],
        jitter=svm_cfg.get("jitter", 0.0),
    )
    kpca = None
    kpca_cfg = config["kpca"]
    if kpca_cfg.get("enabled", False):
        kpca = KernelPCA(
            kernel=build_kernel(kpca_cfg["kernel"]),
            n_components=kpca_cfg.get("n_components"),
            eig_rtol=kpca_cfg.get("eig_rtol", 1e-10),
        )
    return CreditPipeline(svm=svm, kpca=kpca, exclude=excluded_columns(config))


# ---------------------------------------------------------------- baselines
class MajorityClassifier:
    """Trivial baseline: always predict the most frequent class of the training set."""

    def fit(self, X: pd.DataFrame, y) -> MajorityClassifier:
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        self.label_ = labels[np.argmax(counts)]
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        # Un puntaje constante no ordena a los solicitantes: AUC = 0.5.
        return np.zeros(len(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.label_)


class ConstantClassifier(MajorityClassifier):
    """Trivial baseline that always predicts a fixed label (e.g. "reject everyone")."""

    def __init__(self, label: int):
        self.label = label

    def fit(self, X: pd.DataFrame, y) -> ConstantClassifier:
        self.label_ = self.label
        return self


class LogisticBaseline:
    """Logistic regression on the same preprocessed features (reference model).

    The score returned by :meth:`decision_function` is the log-odds of the
    "bad" class, so it is comparable in orientation with the SVM score.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Mapping | str | None = None,
        max_iter: int = 2000,
        exclude: Sequence[str] = (),
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.exclude = tuple(exclude)

    def fit(self, X: pd.DataFrame, y) -> LogisticBaseline:
        self.preprocessor_ = TabularPreprocessor(exclude=self.exclude)
        features = self.preprocessor_.fit_transform(X)
        self.model_ = LogisticRegression(
            C=self.C, class_weight=self.class_weight, max_iter=self.max_iter
        ).fit(features, y)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.decision_function(self.preprocessor_.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.preprocessor_.transform(X))


def build_logistic_baseline(config: Mapping) -> LogisticBaseline:
    """Create the logistic-regression baseline from the configuration."""
    cost = CostMatrix.from_config(config)
    logreg_cfg = config["baselines"]["logistic_regression"]
    return LogisticBaseline(
        C=logreg_cfg["C"],
        class_weight=resolve_class_weight(logreg_cfg.get("class_weight"), cost),
        max_iter=logreg_cfg["max_iter"],
        exclude=excluded_columns(config),
    )


class SklearnSVCReference:
    """``sklearn.svm.SVC`` with the same preprocessing, kernel and hyper-parameters.

    It is used only as a correctness check of the hand-written SVM on the real
    data.  The kernel is passed to scikit-learn as a callable, so both solvers
    work on exactly the same Gram matrix whatever kernel is configured.
    """

    def __init__(self, svm: KernelSVM, exclude: Sequence[str] = ()):
        self.svm = svm
        self.exclude = tuple(exclude)

    def fit(self, X: pd.DataFrame, y) -> SklearnSVCReference:
        self.preprocessor_ = TabularPreprocessor(exclude=self.exclude)
        features = self.preprocessor_.fit_transform(X)
        self.model_ = SVC(
            C=self.svm.C,
            kernel=self.svm.kernel.gram,
            class_weight=self.svm.class_weight,
            tol=self.svm.tol,
        ).fit(features, y)
        return self

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.decision_function(self.preprocessor_.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(self.preprocessor_.transform(X))
