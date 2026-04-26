# ADR 0001 — Planar-Neighbourhood ICP for Boresight Estimation

**Status:** Accepted  
**Date:** 2026-04-26  
**Deciders:** Daud Tasleem

---

## Context

Airborne and UAV LiDAR surveys are collected in multiple overlapping flight strips.
Small errors in the boresight angles (misalignment between the IMU and the LiDAR
scanner head) and lever-arm offsets produce visible seams in the merged point cloud:
adjacent strips are offset by 5–15 cm vertically and show step-like artefacts on
flat surfaces and building rooftops.

No maintained open-source Python library exists to correct this automatically.
The only relevant OSS project (`cdfbdex/StripAdjustment`, 4 ★, last commit July 2022)
uses Coherent Point Drift (CPD) — a method designed for non-rigid registration that
is overkill and slow for this problem, and ships without a public API, tests, or CI.

Commercial tools (BayesStripAlign, LP360, LiDAR360) solve this well but are
proprietary and cost thousands of dollars per seat.

---

## Decision

Use **planar-neighbourhood point-to-plane ICP** minimised by
**Levenberg–Marquardt** over a 6-DOF rigid body model (3 boresight angles + 3
lever-arm components).

### Algorithm sketch

1. **Feature extraction.** For every point, compute PCA on its k-NN neighbourhood.
   Retain patches whose planarity (Weinmann eigenvalue ratio) ≥ threshold.
   Apply normal-space sampling to ensure angular diversity.

2. **Correspondence matching.** Build a KD-tree on reference patch centroids.
   For each target patch, accept the nearest reference patch if the centroid
   distance ≤ `max_dist` AND the inter-normal angle ≤ `max_angle`.

3. **Objective.** Minimise the sum of squared point-to-plane distances:
   `F(p) = Σ_k [ n_k · (R(ω,φ,κ) q_k + t − c_k) ]²`
   where `n_k`, `c_k` are the reference normal and centroid for patch k.

4. **Solver.** `scipy.optimize.least_squares` with `method="lm"` (Levenberg–
   Marquardt), `ftol/xtol/gtol = 1e-9`.

5. **Apply and report.** Transform the target strip by the estimated correction.
   Report per-strip RMSE before/after, number of correspondences, convergence.

---

## Rejected Alternatives

### A. Coherent Point Drift (CPD)

CPD (Myronenko & Song 2010) models registration as a probability density
estimation problem and supports non-rigid deformations. For boresight
calibration, the deformation is strictly rigid (small angle + translation);
CPD's extra flexibility buys nothing and adds 3–10× computational cost.
It also requires choosing a regularisation hyper-parameter that is unintuitive
for practitioners. **Rejected: unnecessarily general, slow.**

### B. Vanilla ICP (point-to-point)

Point-to-point ICP converges to a local minimum on planar surfaces because
it does not exploit surface normals. On a flat rooftop, it cannot distinguish
vertical from horizontal errors. **Rejected: incorrect convergence behaviour
on the dominant geometry type in airborne surveys.**

### C. DEM-based strip adjustment (raster differencing)

Bin points to a raster DEM, subtract, and fit a correction surface.
Simple and fast, but loses sub-pixel precision and cannot handle dense
vegetation or steep slopes where a DEM surface is ill-defined.
(See MDPI Sensors 21 2782, 2021 for a UAV-DEM approach.)
**Rejected: restricted to surveys with clear ground returns.**

### D. RANSAC-based global registration (e.g. Open3D FPFH + RANSAC)

Feature-based global registration (FPFH descriptors + RANSAC) is designed for
scenes with no prior pose estimate. Boresight errors are small (< 0.1°), so
global registration is unnecessary and introduces a factor-of-100 overhead.
**Rejected: solves the wrong problem.**

---

## Consequences

- Correct convergence on flat-terrain surveys (standard airborne and UAV mapping).
- Performance: ~5 s per strip pair at 200k points (see benchmark).
- Limitation: requires some flat or planar surface patches in the overlap zone.
  Surveys over pure dense forest with no ground penetration may yield insufficient
  planar features. Documented in README Limitations section.
- Extension path: add IMU trajectory refinement in v0.2 (see ROADMAP).
