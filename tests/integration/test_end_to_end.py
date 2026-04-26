"""Integration tests: end-to-end happy path on the reference LAS test files."""

from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent.parent / "data"
REF_LAS = DATA / "strip_reference.las"
TGT_LAS = DATA / "strip_target.las"


@pytest.fixture(scope="module")
def las_available() -> bool:
    return REF_LAS.exists() and TGT_LAS.exists()


def test_load_strip_returns_ndarray(las_available: bool) -> None:
    if not las_available:
        pytest.skip("test LAS files not found")
    from lidar_strip_adjust import load_strip

    ref = load_strip(REF_LAS)
    assert ref.ndim == 2
    assert ref.shape[1] == 3
    assert ref.dtype == np.float64
    assert len(ref) > 0


def test_end_to_end_adjustment_reduces_rmse(las_available: bool, tmp_path: Path) -> None:
    """Full pipeline: load → adjust → measure RMSE improvement."""
    if not las_available:
        pytest.skip("test LAS files not found")
    from lidar_strip_adjust import StripAdjuster, compute_strip_rmse, load_strip, save_strip

    reference = load_strip(REF_LAS)
    target = load_strip(TGT_LAS)

    m_before = compute_strip_rmse(reference, target, max_dist=0.5)
    assert m_before["rmse"] > 0.01, "Strips should differ initially"

    adjuster = StripAdjuster(
        k_neighbours=15,
        planarity_threshold=0.55,
        max_patches=5000,
        max_correspondence_dist=2.0,
        max_angle_deg=20.0,
    )
    result = adjuster.adjust(reference, target, strip_id="integration")

    assert result.corrected_points is not None
    assert result.rmse_after < result.rmse_before
    assert result.n_correspondences >= 5

    output_las = tmp_path / "adjusted.las"
    save_strip(result.corrected_points, TGT_LAS, output_las)
    assert output_las.exists()
    assert output_las.stat().st_size > 0

    corrected = load_strip(output_las)
    m_after = compute_strip_rmse(reference, corrected, max_dist=0.5)
    assert m_after["rmse"] < m_before["rmse"], (
        f"RMSE should improve: before={m_before['rmse']:.4f} after={m_after['rmse']:.4f}"
    )
    print(f"\nRMSE: {m_before['rmse']:.4f} → {m_after['rmse']:.4f} m")
    print(
        f"Boresight: ω={np.degrees(result.params[0]):+.4f}° "
        f"φ={np.degrees(result.params[1]):+.4f}° "
        f"κ={np.degrees(result.params[2]):+.4f}°"
    )


def test_cli_runs(las_available: bool, tmp_path: Path) -> None:
    """CLI smoke test using subprocess."""
    if not las_available:
        pytest.skip("test LAS files not found")
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lidar_strip_adjust.cli",
            str(REF_LAS),
            str(TGT_LAS),
            "-o",
            str(tmp_path / "out.las"),
            "--max-patches",
            "3000",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    assert (tmp_path / "out.las").exists()
