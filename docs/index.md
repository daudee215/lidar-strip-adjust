# lidar-strip-adjust

**Automated boresight calibration and strip adjustment for airborne and UAV LiDAR point clouds.**

[![PyPI](https://img.shields.io/pypi/v/lidar-strip-adjust)](https://pypi.org/project/lidar-strip-adjust/)
[![CI](https://github.com/daudee215/lidar-strip-adjust/actions/workflows/ci.yml/badge.svg)](https://github.com/daudee215/lidar-strip-adjust/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/daudee215/lidar-strip-adjust/blob/main/LICENSE)

---

## What it does

`lidar-strip-adjust` estimates a 6-DOF rigid correction (Δω, Δφ, Δκ, Δx, Δy, Δz) between
overlapping flight strips using a planar-neighbourhood point-to-plane ICP algorithm:

1. Extracts planar patches from each strip via k-NN PCA + Weinmann planarity filter
2. Matches patch pairs across strips by proximity and normal agreement
3. Minimises point-to-plane distances with Levenberg–Marquardt optimisation
4. Applies the estimated boresight correction and reports RMSE before/after

## Installation

```bash
pip install lidar-strip-adjust
```

Requires Python ≥ 3.10.

## Quickstart

### CLI

```bash
lidar-strip-adjust reference.las target.las -o corrected.las
```

### Python API

```python
from lidar_strip_adjust import StripAdjuster, load_strip, save_strip

ref = load_strip("reference.las")
tgt = load_strip("target.las")

adjuster = StripAdjuster()
result = adjuster.adjust(ref, tgt, strip_id="strip_02")

print(f"RMSE before: {result.rmse_before:.4f} m")
print(f"RMSE after:  {result.rmse_after:.4f} m")
print(f"Δω={result.params[0]*57.3:.4f}° Δφ={result.params[1]*57.3:.4f}° Δκ={result.params[2]*57.3:.4f}°")

save_strip(result.corrected_points, "target.las", "corrected.las")
```

## Why it exists

No maintained open-source Python package existed for LiDAR boresight calibration as of
early 2026. Practitioners relied on commercial tools (TerraScan, RiPROCESS) or wrote
one-off scripts. See the [Architecture Decision Record](adr/0001-architecture.md) for
algorithm rationale and rejected alternatives.

## Benchmark

On a 200,000-point synthetic dataset (8-core laptop, Python 3.12):

| Metric | Value |
|---|---|
| Wall time | ~5 s |
| Peak memory | ~180 MB |
| RMSE reduction | 91 % |

## Links

- [API Reference](api.md)
- [Architecture Decision](adr/0001-architecture.md)
- [Roadmap](roadmap.md)
- [GitHub](https://github.com/daudee215/lidar-strip-adjust)
