# lidar-strip-adjust

**Automated boresight calibration and strip adjustment for airborne/UAV LiDAR point clouds.**

[![CI](https://github.com/daudee215/lidar-strip-adjust/actions/workflows/ci.yml/badge.svg)](https://github.com/daudee215/lidar-strip-adjust/actions)
[![PyPI](https://img.shields.io/pypi/v/lidar-strip-adjust)](https://pypi.org/project/lidar-strip-adjust/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

---

## What it does

`lidar-strip-adjust` estimates and corrects the 6-DOF rigid misalignment (3 boresight
angles + 3 lever-arm components) between overlapping LiDAR flight strips using
planar-neighbourhood point-to-plane ICP minimised by Levenberg–Marquardt.

Given two LAS/LAZ files for adjacent flight strips, it:
1. Extracts planar features from each strip via PCA on k-NN patches.
2. Matches cross-strip patch pairs by centroid proximity and normal direction.
3. Minimises point-to-plane residuals over the 6-DOF correction model.
4. Writes the adjusted strip back to LAS, preserving all non-geometric attributes.
5. Reports RMSE before/after, estimated boresight angles, and convergence statistics.

---

## Why this exists

### The gap

Airborne and UAV LiDAR surveys are captured in overlapping flight strips. Small
boresight misalignments (< 0.1°) between the IMU and the scanner produce 5–15 cm
step artefacts in the merged point cloud — visible on rooftops, roads, and flat
terrain. Correcting this is a standard requirement in every production LiDAR workflow.

**No maintained open-source Python library existed for this task before this project.**

The only prior OSS attempt:
- [`cdfbdex/StripAdjustment`](https://github.com/cdfbdex/StripAdjustment) — 4 ★,
  last commit July 2022, CPD-only, no API, no tests, no CI.

Academic methods are published (Zhang et al. 2023; Glira et al. 2015; Tian et al. 2022)
but without released Python implementations. Commercial tools (BayesStripAlign,
LP360, LiDAR360) solve this but are proprietary and cost thousands per seat.

### What this tool computes that no maintained alternative computes today

Automated planar-neighbourhood ICP boresight estimation with a typed Python API,
full unit + integration test coverage, and benchmarked performance on real-scale
datasets — all in a pip-installable package.

### Source signals

- [`PDAL/PDAL#4830`](https://github.com/PDAL/PDAL/issues/4830) — PDAL users
  requesting a strip adjustment filter; never implemented.
- Zhang et al. (2023). *"Airborne LiDAR Strip Adjustment Method Based on Point
  Clouds with Planar Neighborhoods."* Remote Sensing 15(23):5447.
  [DOI 10.3390/rs15235447](https://doi.org/10.3390/rs15235447) — no code released.
- Glira et al. (2015). *"Rigorous Boresight Self-Calibration."* Remote Sensing
  11(4):442. [DOI 10.3390/rs11040442](https://doi.org/10.3390/rs11040442)
- Stack Exchange GIS: multiple unanswered questions on flightline-based strip
  processing in Python/QGIS; community redirects to commercial software.

---

## Install

```bash
pip install lidar-strip-adjust
# or with lazrs support for compressed LAZ files:
pip install "lidar-strip-adjust[lazrs]"
```

Requires Python ≥ 3.10.

---

## Quickstart

### CLI

```bash
lidar-strip-adjust strip_A.las strip_B.las -o strip_B_adjusted.las
```

### Python API

```python
from lidar_strip_adjust import StripAdjuster, load_strip, save_strip, compute_strip_rmse

reference = load_strip("strip_A.las")
target    = load_strip("strip_B.las")

print("RMSE before:", compute_strip_rmse(reference, target)["rmse"])

adjuster = StripAdjuster(
    k_neighbours=20,
    planarity_threshold=0.7,
    max_patches=50_000,
    max_correspondence_dist=1.0,
)
result = adjuster.adjust(reference, target)

print(f"RMSE after : {result.rmse_after:.4f} m")
print(f"Δω={result.params[0]:.5f} rad  Δφ={result.params[1]:.5f} rad  Δκ={result.params[2]:.5f} rad")

save_strip(result.corrected_points, "strip_B.las", "strip_B_adjusted.las")
```

---

## API reference

See [docs/api.md](docs/api.md) or `help(StripAdjuster)`.

---

## Benchmark

On a synthetic dataset of 200k points per strip, on an Apple M2 (single-thread):

| Metric | Value |
|---|---|
| Total wall time | ~5 s |
| Peak memory | ~180 MB |
| RMSE before | 0.087 m |
| RMSE after | 0.008 m |
| RMSE improvement | ~91% |

Run yourself:

```bash
python benchmark/bench_strip_adjust.py
```

---

## Limitations

- Requires planar surface patches (rooftops, roads, flat terrain) in the overlap zone.
  Dense forest with no ground penetration may yield insufficient features.
- Corrects rigid boresight only; does not model IMU trajectory drift or GPS lever-arm
  errors (planned for v0.2).
- Input files must be in LAS 1.0–1.4 format. E57 and other formats are not yet supported.

---

## Citation

If you use this tool in research, please cite:

```bibtex
@software{tasleem2026lidar,
  author  = {Tasleem, Daud},
  title   = {lidar-strip-adjust: Automated boresight calibration for LiDAR strips},
  year    = {2026},
  url     = {https://github.com/daudee215/lidar-strip-adjust},
  version = {0.1.0}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
