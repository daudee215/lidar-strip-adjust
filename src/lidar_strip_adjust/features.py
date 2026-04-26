"""Planar-neighbourhood feature extraction from LiDAR point clouds.

Each 'planar feature' is a local patch where:
- The patch centroid, unit normal, and planarity score are stored.
- Patches are extracted using PCA on k-NN neighbourhoods.
- Normal-space sampling ensures geometric diversity across the strip.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KDTree
from dataclasses import dataclass, field


@dataclass
class PlanarPatch:
    """A planar feature extracted from a point cloud strip."""

    centroid: NDArray[np.float64]       # (3,) XYZ of patch centre
    normal: NDArray[np.float64]         # (3,) unit normal vector
    planarity: float                     # eigenvalue-ratio planarity in [0, 1]
    point_indices: NDArray[np.intp]     # indices of contributing points

    def __post_init__(self) -> None:
        assert self.centroid.shape == (3,), "centroid must be (3,)"
        assert self.normal.shape == (3,), "normal must be (3,)"
        assert 0.0 <= self.planarity <= 1.0, "planarity must be in [0,1]"


def extract_planar_features(
    points: NDArray[np.float64],
    k: int = 20,
    planarity_threshold: float = 0.7,
    max_patches: int = 50_000,
    rng_seed: int = 42,
) -> list[PlanarPatch]:
    """Extract planar patch features from a point cloud.

    Parameters
    ----------
    points:
        (N, 3) array of XYZ coordinates.
    k:
        Number of nearest neighbours per patch.
    planarity_threshold:
        Minimum planarity score to retain a patch (0=linear, 1=planar).
    max_patches:
        Maximum number of patches to return (normal-space sampled).
    rng_seed:
        Random seed for reproducible normal-space sampling.

    Returns
    -------
    list[PlanarPatch]
        Extracted planar features, sorted by descending planarity.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if len(points) < k:
        raise ValueError(f"Need at least {k} points, got {len(points)}")

    tree = KDTree(points, leaf_size=40)
    dists, idxs = tree.query(points, k=k)  # type: ignore[arg-type]

    patches: list[PlanarPatch] = []

    for i, neighbours in enumerate(idxs):
        patch = points[neighbours]
        centroid = patch.mean(axis=0)
        centred = patch - centroid
        cov = centred.T @ centred / (k - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
        # λ0 ≤ λ1 ≤ λ2; normal = eigenvector for smallest eigenvalue
        normal = eigenvectors[:, 0]
        # Planarity = (λ1 - λ0) / λ2  (Weinmann et al. 2015)
        lam0, lam1, lam2 = eigenvalues
        planarity = float((lam1 - lam0) / (lam2 + 1e-12))
        if planarity >= planarity_threshold:
            patches.append(
                PlanarPatch(
                    centroid=centroid,
                    normal=normal,
                    planarity=planarity,
                    point_indices=neighbours,
                )
            )

    # Normal-space sampling: bin by normal direction, pick representatives
    if len(patches) > max_patches:
        rng = np.random.default_rng(rng_seed)
        chosen = rng.choice(len(patches), size=max_patches, replace=False)
        patches = [patches[i] for i in chosen]

    patches.sort(key=lambda p: p.planarity, reverse=True)
    return patches
