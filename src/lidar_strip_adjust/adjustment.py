"""Core boresight estimation and strip adjustment via point-to-plane ICP.

The adjustment model
--------------------
A strip j is related to the reference strip i by a rigid transformation
T(Δω, Δφ, Δκ, Δx, Δy, Δz) that encodes boresight angle errors and lever-arm
translation errors between the IMU and the LiDAR scanner head.

The objective minimised is the sum of squared point-to-plane distances:

    F(p) = Σ_k  [ n_k · (R(ω,φ,κ) · q_k + t - c_k) ]²

where:
  - p = (Δω, Δφ, Δκ, Δx, Δy, Δz)  — 6-DOF correction vector
  - q_k  — point in strip j matched to plane k in strip i
  - n_k  — unit normal of plane k
  - c_k  — centroid of plane k
  - R(·) — ZYX rotation matrix from boresight angles

Optimisation uses scipy.optimize.least_squares (Levenberg–Marquardt variant).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from lidar_strip_adjust.features import PlanarPatch, extract_planar_features

logger = logging.getLogger(__name__)


# ── Rotation helpers ────────────────────────────────────────────────────────


def _zyx_rotation(omega: float, phi: float, kappa: float) -> NDArray[np.float64]:
    """ZYX (yaw-pitch-roll) rotation matrix from boresight angles (radians)."""
    co, so = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi), np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)
    Rz = np.array([[ck, -sk, 0], [sk, ck, 0], [0, 0, 1]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, co, -so], [0, so, co]], dtype=np.float64)
    return Rz @ Ry @ Rx


def apply_correction(
    points: NDArray[np.float64],
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a 6-DOF rigid correction to a point cloud.

    Parameters
    ----------
    points:
        (N, 3) input point cloud.
    params:
        (6,) array [Δω, Δφ, Δκ, Δx, Δy, Δz] in radians / same unit as points.

    Returns
    -------
    NDArray[np.float64]
        (N, 3) corrected point cloud.
    """
    omega, phi, kappa = params[:3]
    translation = params[3:]
    R = _zyx_rotation(omega, phi, kappa)
    return (R @ points.T).T + translation


# ── Correspondence matching ─────────────────────────────────────────────────


def _match_patches(
    ref_patches: list[PlanarPatch],
    tgt_patches: list[PlanarPatch],
    max_dist: float = 1.0,
    max_angle_deg: float = 15.0,
) -> list[tuple[PlanarPatch, PlanarPatch]]:
    """Match planar patches between reference and target strips.

    A pair is accepted when the centroid distance ≤ max_dist AND the angle
    between normals ≤ max_angle_deg (or its supplement ≤ max_angle_deg).
    """
    ref_centroids = np.array([p.centroid for p in ref_patches])
    tgt_centroids = np.array([p.centroid for p in tgt_patches])
    tree = cKDTree(ref_centroids)
    dists, idxs = tree.query(tgt_centroids, k=1, workers=-1)

    cos_thresh = np.cos(np.radians(max_angle_deg))
    pairs: list[tuple[PlanarPatch, PlanarPatch]] = []
    for j, (dist, i) in enumerate(zip(dists, idxs, strict=False)):
        if dist > max_dist:
            continue
        rp, tp = ref_patches[i], tgt_patches[j]
        cos_angle = abs(float(np.dot(rp.normal, tp.normal)))
        if cos_angle >= cos_thresh:
            pairs.append((rp, tp))
    logger.debug("Matched %d/%d patch pairs", len(pairs), len(tgt_patches))
    return pairs


# ── Residual function ───────────────────────────────────────────────────────


def _residuals(
    params: NDArray[np.float64],
    pairs: list[tuple[PlanarPatch, PlanarPatch]],
    tgt_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Point-to-plane residuals for all matched pairs."""
    omega, phi, kappa = params[:3]
    t = params[3:]
    R = _zyx_rotation(omega, phi, kappa)
    res = []
    for ref_patch, tgt_patch in pairs:
        # Transform patch centroid of target to reference frame
        q_corr = R @ tgt_patch.centroid + t
        dist = float(ref_patch.normal @ (q_corr - ref_patch.centroid))
        res.append(dist)
    return np.array(res, dtype=np.float64)


# ── Main adjuster ───────────────────────────────────────────────────────────


@dataclass
class AdjustmentResult:
    """Result of a strip-pair adjustment."""

    strip_id: str
    params: NDArray[np.float64]  # (6,) [Δω,Δφ,Δκ,Δx,Δy,Δz]
    rmse_before: float  # RMS point-to-plane before adjustment
    rmse_after: float  # RMS point-to-plane after adjustment
    n_correspondences: int  # number of matched patch pairs
    converged: bool
    n_iterations: int
    cost: float
    corrected_points: NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 3)),
        repr=False,
    )


class StripAdjuster:
    """Boresight calibration and strip adjustment.

    Parameters
    ----------
    k_neighbours:
        k for local PCA neighbourhood (planar feature extraction).
    planarity_threshold:
        Minimum planarity score for a patch to be retained.
    max_patches:
        Maximum patches per strip (normal-space sampled).
    max_correspondence_dist:
        Maximum centroid distance (same unit as coordinates) for patch matching.
    max_angle_deg:
        Maximum inter-normal angle for patch matching (degrees).
    max_iterations:
        Iteration limit passed to scipy LM solver.
    """

    def __init__(
        self,
        k_neighbours: int = 20,
        planarity_threshold: float = 0.7,
        max_patches: int = 50_000,
        max_correspondence_dist: float = 1.0,
        max_angle_deg: float = 15.0,
        max_iterations: int = 200,
    ) -> None:
        self.k_neighbours = k_neighbours
        self.planarity_threshold = planarity_threshold
        self.max_patches = max_patches
        self.max_correspondence_dist = max_correspondence_dist
        self.max_angle_deg = max_angle_deg
        self.max_iterations = max_iterations

    def adjust(
        self,
        reference: NDArray[np.float64],
        target: NDArray[np.float64],
        strip_id: str = "target",
        x0: NDArray[np.float64] | None = None,
    ) -> AdjustmentResult:
        """Estimate and apply the 6-DOF boresight correction.

        Parameters
        ----------
        reference:
            (N, 3) reference strip (kept fixed).
        target:
            (M, 3) target strip to be corrected.
        strip_id:
            Identifier for logging and reporting.
        x0:
            Initial parameter guess [Δω,Δφ,Δκ,Δx,Δy,Δz]; defaults to zeros.

        Returns
        -------
        AdjustmentResult
        """
        if x0 is None:
            x0 = np.zeros(6, dtype=np.float64)

        logger.info("Extracting features from reference strip (%d pts)", len(reference))
        ref_patches = extract_planar_features(
            reference,
            k=self.k_neighbours,
            planarity_threshold=self.planarity_threshold,
            max_patches=self.max_patches,
        )
        logger.info("Extracted %d reference patches", len(ref_patches))

        logger.info("Extracting features from target strip (%d pts)", len(target))
        tgt_patches = extract_planar_features(
            target,
            k=self.k_neighbours,
            planarity_threshold=self.planarity_threshold,
            max_patches=self.max_patches,
        )
        logger.info("Extracted %d target patches", len(tgt_patches))

        if not ref_patches or not tgt_patches:
            raise ValueError(
                "Insufficient planar features extracted. "
                "Try lowering planarity_threshold or k_neighbours."
            )

        pairs = _match_patches(
            ref_patches,
            tgt_patches,
            max_dist=self.max_correspondence_dist,
            max_angle_deg=self.max_angle_deg,
        )
        if len(pairs) < 10:
            raise ValueError(
                f"Only {len(pairs)} patch correspondences found (need ≥ 10). "
                "Increase max_correspondence_dist or lower planarity_threshold."
            )

        # RMS before adjustment
        res_before = _residuals(x0, pairs, target)
        rmse_before = float(np.sqrt(np.mean(res_before**2)))

        result = least_squares(
            _residuals,
            x0,
            args=(pairs, target),
            method="lm",
            max_nfev=self.max_iterations * len(x0),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        params = result.x
        converged = result.success or result.cost < 1e-6

        res_after = _residuals(params, pairs, target)
        rmse_after = float(np.sqrt(np.mean(res_after**2)))

        corrected = apply_correction(target, params)

        logger.info(
            "Strip %s: RMSE %.4f → %.4f m (Δω=%.4f° Δφ=%.4f° Δκ=%.4f°)",
            strip_id,
            rmse_before,
            rmse_after,
            np.degrees(params[0]),
            np.degrees(params[1]),
            np.degrees(params[2]),
        )

        return AdjustmentResult(
            strip_id=strip_id,
            params=params,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            n_correspondences=len(pairs),
            converged=converged,
            n_iterations=result.nfev,
            cost=float(result.cost),
            corrected_points=corrected,
        )
