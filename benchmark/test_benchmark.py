"""pytest-benchmark wrapper for CI/CD benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

N_BENCH = 50_000  # smaller for CI speed; full bench uses 200k


def _strips(n: int = N_BENCH):
    from lidar_strip_adjust.adjustment import _zyx_rotation

    rng = np.random.default_rng(77)
    flat_xy = rng.uniform(0, 200, (n, 2))
    flat_z = 100.0 + rng.normal(0, 0.01, n)
    ref = np.column_stack([flat_xy, flat_z])
    R = _zyx_rotation(np.radians(0.05), np.radians(0.02), np.radians(0.03))
    tgt = (R @ ref.T).T + np.array([0, 0, 0.05]) + rng.normal(0, 0.005, ref.shape)
    return ref, tgt


def test_feature_extraction_speed(benchmark):
    from lidar_strip_adjust.features import extract_planar_features

    ref, _ = _strips()
    result = benchmark(
        extract_planar_features, ref, k=15, planarity_threshold=0.6, max_patches=10_000
    )
    assert len(result) > 0


def test_full_adjustment_speed(benchmark):
    from lidar_strip_adjust.adjustment import StripAdjuster

    ref, tgt = _strips()
    adjuster = StripAdjuster(k_neighbours=15, planarity_threshold=0.6, max_patches=10_000)
    result = benchmark(adjuster.adjust, ref, tgt)
    assert result.rmse_after < result.rmse_before
