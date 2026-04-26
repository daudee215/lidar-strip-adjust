"""Unit tests for planar feature extraction."""

import numpy as np
import pytest

from lidar_strip_adjust.features import PlanarPatch, extract_planar_features


def _flat_plane(n: int = 500, noise: float = 0.005, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 10, (n, 2))
    z = 5.0 + rng.normal(0, noise, n)
    return np.column_stack([xy, z])


def test_returns_list_of_patches():
    pts = _flat_plane(500)
    patches = extract_planar_features(pts, k=15, planarity_threshold=0.5, max_patches=200)
    assert isinstance(patches, list)
    assert len(patches) > 0
    assert all(isinstance(p, PlanarPatch) for p in patches)


def test_planar_patches_have_high_planarity_on_flat_data():
    pts = _flat_plane(800, noise=0.001)
    patches = extract_planar_features(pts, k=20, planarity_threshold=0.8)
    assert len(patches) > 0
    planarity_values = [p.planarity for p in patches]
    assert min(planarity_values) >= 0.8


def test_normals_are_unit_vectors():
    pts = _flat_plane(600)
    patches = extract_planar_features(pts, k=15, planarity_threshold=0.5)
    for p in patches:
        assert abs(np.linalg.norm(p.normal) - 1.0) < 1e-10


def test_normals_approximately_vertical_for_flat_plane():
    """On a horizontal plane, normals should be ~vertical."""
    pts = _flat_plane(600, noise=0.001)
    patches = extract_planar_features(pts, k=20, planarity_threshold=0.7)
    for p in patches:
        assert abs(p.normal[2]) > 0.9, f"Normal not vertical: {p.normal}"


def test_raises_on_insufficient_points():
    pts = np.random.rand(5, 3)
    with pytest.raises(ValueError, match="at least"):
        extract_planar_features(pts, k=20)


def test_raises_on_wrong_shape():
    pts = np.random.rand(100, 2)
    with pytest.raises(ValueError, match="\\(N, 3\\)"):
        extract_planar_features(pts)


def test_max_patches_limits_output():
    pts = _flat_plane(1000)
    patches = extract_planar_features(pts, k=10, planarity_threshold=0.1, max_patches=50)
    assert len(patches) <= 50


def test_sorted_descending_planarity():
    pts = _flat_plane(600)
    patches = extract_planar_features(pts, k=15, planarity_threshold=0.3)
    for a, b in zip(patches, patches[1:]):
        assert a.planarity >= b.planarity
