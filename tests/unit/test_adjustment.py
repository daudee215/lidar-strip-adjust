"""Unit tests for the core StripAdjuster."""

import numpy as np
import pytest

from lidar_strip_adjust.adjustment import (
    AdjustmentResult,
    StripAdjuster,
    _zyx_rotation,
    apply_correction,
)


def _make_synthetic_strips(
    n: int = 2000,
    omega_deg: float = 0.05,
    phi_deg: float = 0.02,
    kappa_deg: float = 0.03,
    tz: float = 0.05,
    noise: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reference strip and a perturbed target strip."""
    rng = np.random.default_rng(seed)
    # Flat plane + some wall patches for rich planarity
    flat_xy = rng.uniform(0, 40, (n - n // 5, 2))
    flat_z = 100.0 + rng.normal(0, 0.01, n - n // 5)
    wall_x = rng.uniform(5, 7, n // 5)
    wall_y = rng.uniform(0, 20, n // 5)
    wall_z = rng.uniform(95, 110, n // 5)
    ref = np.vstack([np.column_stack([flat_xy, flat_z]), np.column_stack([wall_x, wall_y, wall_z])])
    R = _zyx_rotation(
        np.radians(omega_deg),
        np.radians(phi_deg),
        np.radians(kappa_deg),
    )
    t = np.array([0.0, 0.0, tz])
    tgt = (R @ ref.T).T + t + rng.normal(0, noise, ref.shape)
    return ref, tgt


def test_rotation_matrix_is_orthonormal():
    for angles in [(0, 0, 0), (0.1, 0.2, 0.3), (-0.05, 0.01, -0.02)]:
        R = _zyx_rotation(*angles)
        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


def test_apply_correction_identity():
    pts = np.random.rand(100, 3)
    corrected = apply_correction(pts, np.zeros(6))
    assert np.allclose(corrected, pts, atol=1e-12)


def test_apply_correction_known_translation():
    pts = np.ones((10, 3))
    params = np.array([0, 0, 0, 1.0, 2.0, 3.0])
    corrected = apply_correction(pts, params)
    assert np.allclose(corrected, pts + np.array([1, 2, 3]), atol=1e-10)


def test_adjuster_reduces_rmse():
    ref, tgt = _make_synthetic_strips(2000, seed=42)
    adjuster = StripAdjuster(
        k_neighbours=15,
        planarity_threshold=0.6,
        max_patches=5000,
        max_correspondence_dist=2.0,
    )
    result = adjuster.adjust(ref, tgt, strip_id="test")
    assert isinstance(result, AdjustmentResult)
    assert result.rmse_after < result.rmse_before
    assert result.n_correspondences >= 10


def test_adjuster_recovers_boresight_angles():
    """Adjuster returns the *correction* (inverse of applied error).

    The target is created by applying R(+omega, +phi, +kappa) + t to the
    reference, so the adjuster should return params ~= (-omega, -phi, -kappa, ...)
    to undo that error.  We verify the magnitude of recovery within 0.02 deg.
    """
    true_omega, true_phi, true_kappa = 0.05, 0.02, 0.03
    ref, tgt = _make_synthetic_strips(
        3000, true_omega, true_phi, true_kappa, tz=0.05, noise=0.002, seed=7
    )
    adjuster = StripAdjuster(
        k_neighbours=20,
        planarity_threshold=0.65,
        max_patches=8000,
        max_correspondence_dist=2.0,
    )
    result = adjuster.adjust(ref, tgt)
    est_omega = np.degrees(result.params[0])
    est_phi = np.degrees(result.params[1])
    # params are the correction: should be approx -applied_error
    assert abs(est_omega + true_omega) < 0.02, (
        f"omega recovery error: {abs(est_omega + true_omega):.4f} deg "
        f"(est={est_omega:.4f}, applied={true_omega:.4f})"
    )
    assert abs(est_phi + true_phi) < 0.02, (
        f"phi recovery error: {abs(est_phi + true_phi):.4f} deg "
        f"(est={est_phi:.4f}, applied={true_phi:.4f})"
    )


def test_adjuster_raises_on_too_few_points():
    ref = np.random.rand(10, 3)
    tgt = np.random.rand(10, 3)
    adjuster = StripAdjuster(k_neighbours=20)
    with pytest.raises(ValueError):
        adjuster.adjust(ref, tgt)
