# API Reference

## Module: `lidar_strip_adjust`

Top-level re-exports:

```python
from lidar_strip_adjust import (
    StripAdjuster,
    load_strip,
    save_strip,
    extract_planar_features,
    compute_strip_rmse,
)
```

---

## I/O

### `load_strip`

```python
def load_strip(path: str | Path, scale_to_metres: bool = True) -> NDArray[np.float64]
```

Load a LAS/LAZ file and return an `(N, 3)` float64 array of XYZ coordinates.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | Path to `.las` or `.laz` file |
| `scale_to_metres` | `bool` | `True` | Apply LAS scale/offset to convert integer coords to metres |

**Returns** `NDArray[np.float64]` — shape `(N, 3)`.

**Raises** `FileNotFoundError` if the file does not exist.

---

### `save_strip`

```python
def save_strip(
    points: NDArray[np.float64],
    template_path: str | Path,
    output_path: str | Path,
) -> None
```

Write corrected XYZ coordinates back to a LAS file, preserving all non-XYZ attributes
(intensity, classification, return number, GPS time, etc.) from the template.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `points` | `NDArray[np.float64]` | `(N, 3)` corrected XYZ |
| `template_path` | `str \| Path` | Source LAS file to copy attributes from |
| `output_path` | `str \| Path` | Destination `.las` file |

---

## Feature extraction

### `extract_planar_features`

```python
def extract_planar_features(
    points: NDArray[np.float64],
    k: int = 20,
    planarity_threshold: float = 0.7,
    max_patches: int = 50_000,
    rng_seed: int = 42,
) -> list[PlanarPatch]
```

Extract planar patches from a point cloud using k-NN PCA and Weinmann planarity filter.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `points` | `NDArray[np.float64]` | — | `(N, 3)` point cloud |
| `k` | `int` | `20` | Neighbourhood size for PCA |
| `planarity_threshold` | `float` | `0.7` | Minimum `(λ1−λ0)/λ2` Weinmann ratio |
| `max_patches` | `int` | `50_000` | Subsample if more patches found |
| `rng_seed` | `int` | `42` | RNG seed for subsampling |

**Returns** `list[PlanarPatch]` sorted descending by planarity.

### `PlanarPatch`

```python
@dataclass
class PlanarPatch:
    centroid: NDArray[np.float64]    # (3,) — patch centre
    normal: NDArray[np.float64]      # (3,) — unit normal (PCA min-eigenvalue axis)
    planarity: float                  # Weinmann ratio in [0, 1]
    point_indices: NDArray[np.intp]  # indices into original point array
```

---

## Adjustment

### `StripAdjuster`

```python
class StripAdjuster:
    def __init__(
        self,
        k: int = 20,
        planarity_threshold: float = 0.7,
        max_patches: int = 50_000,
        max_dist: float = 1.0,
        max_angle_deg: float = 15.0,
    ) -> None
```

Estimates and applies a 6-DOF boresight correction between two overlapping LiDAR strips.

#### `adjust`

```python
def adjust(
    self,
    reference: NDArray[np.float64],
    target: NDArray[np.float64],
    strip_id: str = "target",
    x0: NDArray[np.float64] | None = None,
) -> AdjustmentResult
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `reference` | `NDArray[np.float64]` | — | `(N, 3)` reference strip (fixed) |
| `target` | `NDArray[np.float64]` | — | `(M, 3)` target strip to correct |
| `strip_id` | `str` | `"target"` | Label for logging |
| `x0` | `NDArray[np.float64] \| None` | `None` | Initial params `[ω, φ, κ, tx, ty, tz]` (default zeros) |

**Returns** `AdjustmentResult`.

### `AdjustmentResult`

```python
@dataclass
class AdjustmentResult:
    strip_id: str
    params: NDArray[np.float64]       # [ω, φ, κ, tx, ty, tz] in radians + metres
    corrected_points: NDArray[np.float64]
    rmse_before: float
    rmse_after: float
    n_correspondences: int
    converged: bool
    cost: float
```

---

## Metrics

### `compute_strip_rmse`

```python
def compute_strip_rmse(
    reference: NDArray[np.float64],
    target: NDArray[np.float64],
    max_dist: float = 0.5,
    subsample: int = 100_000,
) -> dict[str, float]
```

Compute RMSE, MAE, and coverage between two strips using nearest-neighbour matching.

**Returns** dict with keys: `rmse`, `mae`, `n_pairs`, `coverage_fraction`.

---

## CLI

```
lidar-strip-adjust [-h] [-o OUTPUT] [--k K] [--planarity P] [--max-patches N]
                   [--max-dist D] [--max-angle A] [-v]
                   reference target
```

| Argument | Description |
|---|---|
| `reference` | Reference LAS/LAZ strip (fixed) |
| `target` | Target LAS/LAZ strip to adjust |
| `-o OUTPUT` | Output path for corrected strip (default: `<target>_adjusted.las`) |
| `--k` | k-NN neighbourhood size (default: 20) |
| `--planarity` | Planarity threshold (default: 0.7) |
| `--max-patches` | Max patches per strip (default: 50,000) |
| `--max-dist` | Max match distance in metres (default: 1.0) |
| `--max-angle` | Max normal angle in degrees (default: 15.0) |
| `-v` | Verbose output |
