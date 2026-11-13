"""Numerical helpers: centring, symmetrisation, jitter, spectrum clipping."""

from __future__ import annotations

import numpy as np
import pytest

from rkhs_credit.kernels import RBFKernel
from rkhs_credit.numerics import (
    add_jitter,
    center_gram,
    center_gram_test,
    clip_spectrum,
    spectral_summary,
    symmetrize,
)


def centering_matrix(n: int) -> np.ndarray:
    return np.eye(n) - np.ones((n, n)) / n


def test_centered_gram_has_zero_mean_rows_and_columns(rng):
    X = rng.normal(size=(50, 4))
    K_centered = center_gram(RBFKernel(0.5).gram(X))
    scale = np.abs(K_centered).max()
    assert np.abs(K_centered.mean(axis=0)).max() <= 1e-12 * scale
    assert np.abs(K_centered.mean(axis=1)).max() <= 1e-12 * scale
    assert np.array_equal(K_centered, K_centered.T)


def test_center_gram_equals_explicit_HKH(rng):
    X = rng.normal(size=(30, 3))
    K = RBFKernel(0.2).gram(X)
    H = centering_matrix(30)
    np.testing.assert_allclose(center_gram(K), H @ K @ H, atol=1e-12)


def test_centering_matrix_is_an_orthogonal_projector():
    """H = H^T = H^2 and H 1 = 0 (the facts behind the centring formula)."""
    H = centering_matrix(8)
    np.testing.assert_allclose(H, H.T)
    np.testing.assert_allclose(H @ H, H, atol=1e-14)
    np.testing.assert_allclose(H @ np.ones(8), 0.0, atol=1e-14)
    eigenvalues = np.sort(np.linalg.eigvalsh(H))
    np.testing.assert_allclose(eigenvalues, [0.0] + [1.0] * 7, atol=1e-12)


def test_center_gram_test_matches_explicit_feature_centering(rng):
    """With the linear kernel the feature map is explicit, so centring can be checked."""
    X = rng.normal(size=(25, 4))
    X_new = rng.normal(size=(10, 4))
    mean = X.mean(axis=0)
    expected = (X_new - mean) @ (X - mean).T
    np.testing.assert_allclose(center_gram_test(X_new @ X.T, X @ X.T), expected, atol=1e-11)


def test_center_gram_test_on_training_points_recovers_center_gram(rng):
    X = rng.normal(size=(20, 3))
    K = RBFKernel(1.0).gram(X)
    np.testing.assert_allclose(center_gram_test(K, K), center_gram(K), atol=1e-12)


def test_symmetrize_and_add_jitter(rng):
    A = rng.normal(size=(6, 6))
    S = symmetrize(A)
    assert np.array_equal(S, S.T)
    np.testing.assert_allclose(S, 0.5 * (A + A.T))
    with pytest.raises(ValueError):
        symmetrize(rng.normal(size=(3, 4)))

    K = np.eye(3)
    K_jittered = add_jitter(K, 1e-3)
    np.testing.assert_allclose(K_jittered, (1 + 1e-3) * np.eye(3))
    assert np.array_equal(K, np.eye(3))  # el original no se modifica
    with pytest.raises(ValueError):
        add_jitter(K, -1.0)


def test_clip_spectrum_zeroes_roundoff_and_warns_on_real_negatives():
    eigenvalues = np.array([10.0, 1.0, 5e-13, -3e-14, -2e-12])
    clipped = clip_spectrum(eigenvalues, rtol=1e-10)
    np.testing.assert_array_equal(clipped, [10.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.warns(RuntimeWarning, match="not positive semidefinite"):
        clipped = clip_spectrum(np.array([10.0, 1.0, -0.5]))
    np.testing.assert_array_equal(clipped, [10.0, 1.0, 0.0])


def test_spectral_summary_condition_numbers():
    K = np.diag([100.0, 4.0, 1e-3])
    summary = spectral_summary(K, jitter=1.0)
    assert summary.condition_number == pytest.approx(1e5)
    assert summary.numerical_rank == 3
    # kappa(K + eps I) = (lambda_max + eps) / (lambda_min + eps) <= 1 + lambda_max / eps
    assert summary.condition_number_with_jitter == pytest.approx(101.0 / 1.001)
    assert summary.condition_number_with_jitter <= 1 + 100.0 / 1.0

    singular = spectral_summary(np.diag([5.0, 1.0, 0.0]), jitter=0.5)
    assert singular.condition_number == float("inf")
    assert singular.effective_condition_number == pytest.approx(5.0)
    assert singular.numerical_rank == 2
    assert singular.condition_number_with_jitter == pytest.approx(5.5 / 0.5)


def test_jitter_improves_conditioning_of_a_gram_matrix(rng):
    X = rng.normal(size=(40, 2))
    K = RBFKernel(0.01).gram(X)  # kernel muy ancho: matriz casi singular
    summary = spectral_summary(K, jitter=1e-6)
    assert summary.condition_number_with_jitter < summary.condition_number
    assert summary.condition_number_with_jitter <= 1 + summary.lambda_max / 1e-6
