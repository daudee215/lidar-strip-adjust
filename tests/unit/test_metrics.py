"""Unit tests for strip RMSE metrics."""

import numpy as np
import pytest

from lidar_strip_adjust.metrics import compute_strip_rmse


def test_identical_strips_give_zero_rmse():
    pts = np.random.default_rng(0).uniform(0, 10, (500, 3))
    m = compute_strip_rmse(pts, pts, max_dist=1.0)
    assert m["rmse"] < 1e-6
    assert m["n_pairs"] == 500


def test_shifted_strip_gives_expected_rmse():
    rng = np.random.default_rng(1)
    ref = rng.uniform(0, 10, (1000, 3))
    shift = 0.2
    tgt = ref + np.array([0, 0, shift])
    m = compute_strip_rmse(ref, tgt, max_dist=0.5)
    assert abs(m["rmse"] - shift) < 0.01
    assert m["coverage_fraction"] > 0.9


def test_non_overlapping_returns_inf():
    ref = np.zeros((100, 3))
    tgt = np.ones((100, 3)) * 100.0
    m = compute_strip_rmse(ref, tgt, max_dist=0.5)
    assert m["rmse"] == np.inf
    assert m["n_pairs"] == 0


def test_subsampling_preserves_rmse():
    rng = np.random.default_rng(2)
    ref = rng.uniform(0, 10, (5000, 3))
    tgt = ref + np.array([0, 0, 0.1])
    m_full = compute_strip_rmse(ref, tgt, subsample=5000)
    m_sub = compute_strip_rmse(ref, tgt, subsample=500)
    assert abs(m_full["rmse"] - m_sub["rmse"]) < 0.02
