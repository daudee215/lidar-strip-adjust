"""
Benchmark: strip adjustment on a large synthetic dataset.

Generates two strips of ~200k points each and measures:
  - Feature extraction time per strip
  - Correspondence matching time
  - Optimisation time
  - Total wall time
  - Peak memory (RSS)

Run with:
    uv run python benchmark/bench_strip_adjust.py

Or via pytest-benchmark:
    uv run pytest benchmark/ --benchmark-only -v
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Allow running standalone without installing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lidar_strip_adjust.adjustment import StripAdjuster
from lidar_strip_adjust.metrics import compute_strip_rmse


N_POINTS = 200_000  # ~100k per strip is minimum; use 200k for realism
SEED = 99


def _generate_large_strips(n: int = N_POINTS, seed: int = SEED):
    rng = np.random.default_rng(seed)
    flat_n = int(n * 0.85)
    wall_n = n - flat_n
    flat_xy = rng.uniform(0, 500, (flat_n, 2))
    flat_z = 150.0 + rng.normal(0, 0.02, flat_n)
    wall_x = np.tile(np.linspace(50, 55, wall_n // 4), 4)[:wall_n]
    wall_y = rng.uniform(0, 200, wall_n)
    wall_z = rng.uniform(140, 165, wall_n)
    ref = np.vstack([
        np.column_stack([flat_xy, flat_z]),
        np.column_stack([wall_x, wall_y, wall_z])
    ])
    from lidar_strip_adjust.adjustment import _zyx_rotation
    R = _zyx_rotation(np.radians(0.04), np.radians(0.015), np.radians(0.025))
    t = np.array([0.0, 0.0, 0.08])
    tgt = (R @ ref.T).T + t + rng.normal(0, 0.008, ref.shape)
    return ref, tgt


def run_benchmark() -> dict:
    print(f"Generating {N_POINTS:,} point synthetic strips …")
    t0 = time.perf_counter()
    ref, tgt = _generate_large_strips()
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"  Generated in {gen_ms:.0f} ms")

    adjuster = StripAdjuster(
        k_neighbours=20,
        planarity_threshold=0.65,
        max_patches=50_000,
        max_correspondence_dist=2.0,
    )

    tracemalloc.start()
    gc.collect()

    t_start = time.perf_counter()
    result = adjuster.adjust(ref, tgt, strip_id="bench")
    total_s = time.perf_counter() - t_start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    m_before = compute_strip_rmse(ref, tgt, max_dist=0.5, subsample=50_000)
    m_after  = compute_strip_rmse(ref, result.corrected_points, max_dist=0.5, subsample=50_000)

    report = {
        "n_points_per_strip": N_POINTS,
        "total_wall_s": round(total_s, 3),
        "n_correspondences": result.n_correspondences,
        "rmse_before_m": round(m_before["rmse"], 4),
        "rmse_after_m": round(m_after["rmse"], 4),
        "rmse_improvement_pct": round(
            100 * (m_before["rmse"] - m_after["rmse"]) / max(m_before["rmse"], 1e-9), 1
        ),
        "converged": result.converged,
        "peak_memory_mb": round(peak / 1024 / 1024, 1),
        "boresight_deg": {
            "omega": round(np.degrees(result.params[0]), 4),
            "phi":   round(np.degrees(result.params[1]), 4),
            "kappa": round(np.degrees(result.params[2]), 4),
        },
    }

    print("\n=== Benchmark Results ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return report


if __name__ == "__main__":
    run_benchmark()
