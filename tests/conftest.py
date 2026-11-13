"""Shared fixtures.  Every test uses small synthetic data: no network access."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rkhs_credit.config import load_config, with_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def two_class_data(rng):
    """Binary problem with a linear and a non-linear component plus noise (n = 200, d = 5).

    The linear part is strong on purpose: if it were weak, the weighted *linear*
    SVM would collapse to a constant function (w = 0), and comparing two
    constant decision functions by correlation would be meaningless.
    """
    X = rng.normal(size=(200, 5))
    signal = X[:, 0] * X[:, 1] + 1.5 * X[:, 2] + 0.4 * rng.normal(size=200)
    y = (signal > 0.2).astype(int)
    X_new = rng.normal(size=(150, 5))
    return X, y, X_new


def make_credit_like_frame(n: int = 180, seed: int = 7) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic table with the same kinds of columns as German Credit."""
    rng = np.random.default_rng(seed)
    age = rng.integers(19, 70, size=n).astype(float)
    duration = rng.integers(6, 60, size=n).astype(float)
    credit_amount = np.round(rng.lognormal(mean=8.0, sigma=0.8, size=n))
    checking_status = rng.choice(["<0", "0<=X<200", "no checking"], size=n)
    purpose = rng.choice(["new car", "radio/tv", "education", "business"], size=n)
    personal_status = rng.choice(
        ["male single", "female div/dep/mar", "male mar/wid"], size=n, p=[0.5, 0.35, 0.15]
    )
    foreign_worker = rng.choice(["yes", "no"], size=n, p=[0.85, 0.15])
    risk = 0.04 * duration + 1.0 * (checking_status == "<0") - 0.03 * age + 0.6 * rng.normal(size=n)
    y = (risk > np.quantile(risk, 0.7)).astype(int)  # ~30 % "bad", como en German Credit
    X = pd.DataFrame(
        {
            "checking_status": checking_status,
            "duration": duration,
            "purpose": purpose,
            "credit_amount": credit_amount,
            "personal_status": personal_status,
            "age": age,
            "foreign_worker": foreign_worker,
        }
    )
    return X, y


@pytest.fixture
def credit_like_data() -> tuple[pd.DataFrame, np.ndarray]:
    return make_credit_like_frame()


@pytest.fixture
def small_config(tmp_path) -> dict:
    """The default configuration with tiny grids and outputs under ``tmp_path``."""
    return with_overrides(
        load_config(DEFAULT_CONFIG),
        {
            "paths.data_dir": str(tmp_path / "data"),
            "paths.results_dir": str(tmp_path / "results"),
            "paths.figures_dir": str(tmp_path / "figures"),
            "cv.n_splits": 3,
            "cv.grid": {"svm.C": [0.3, 3.0], "svm.kernel.gamma": [0.01, 0.1]},
            "baselines.logistic_regression.grid": {"baselines.logistic_regression.C": [0.1, 1.0]},
            "variants.kpca_svm.overrides": {
                "kpca.enabled": True,
                "kpca.n_components": 5,
                "svm.kernel": {"name": "linear"},
            },
            "variants.kpca_svm.grid": {"svm.C": [0.3, 3.0]},
            "figures.formats": ["png"],
            "figures.dpi": 50,
            "figures.spectrum_components": 30,
            "figures.boundary_grid_size": 40,
            "figures.boundary_gamma": 2.0,
            "figures.boundary_C": 1.0,
        },
    )
