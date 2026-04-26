# Roadmap

## v0.1.0 — Initial Release (current)

**Goal:** Working CLI and Python API for single-pair boresight calibration.

- Planar-neighbourhood feature extraction (k-NN PCA + Weinmann planarity filter)
- Point-to-plane ICP with Levenberg–Marquardt optimisation (6-DOF: Δω, Δφ, Δκ, Δx, Δy, Δz)
- LAS/LAZ I/O via laspy (preserving all non-XYZ attributes)
- RMSE/MAE metrics with coverage fraction reporting
- CLI: `lidar-strip-adjust reference.las target.las -o corrected.las`
- Unit + integration tests, CI via GitHub Actions
- MIT licence, CITATION.cff, mkdocs documentation

## v0.2.0 — Multi-strip Network Adjustment

**Goal:** Simultaneous least-squares adjustment across N ≥ 3 flight strips.

- Strip graph construction: pair all overlapping strip combinations automatically
- Weighted global normal-equations solver (scipy sparse + LSQR)
- Per-strip residual report and blunder detection (3σ rejection)
- Tie-point / control-point injection (optional ground-truth LAS)
- Parallelised pairwise feature extraction (`concurrent.futures`)
- `--network` CLI subcommand for batch input
- Benchmark: ≥10 strips, ≥5M points each, wall time < 60 s on 8 cores

## v1.0.0 — Production-Ready SDK

**Goal:** API stability, cloud-native support, and ecosystem integrations.

- Stable public API with semantic-versioning guarantees
- PDAL pipeline integration: `lidar_strip_adjust` filter stage (JSON pipeline)
- Cloud I/O: direct read/write from S3, Azure Blob, GCS via `fsspec`
- LAZ compression support via lazrs/laszip backend
- Per-point confidence scores written back to LAS extra bytes
- Full type stubs (`py.typed`, mypy strict)
- Sphinx/ReadTheDocs hosted documentation
- Conda-forge package
