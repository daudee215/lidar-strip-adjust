"""Strip adjustment quality metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def compute_strip_rmse(
    reference: NDArray[np.float64],
    target: NDArray[np.float64],
    max_dist: float = 0.5,
    subsample: int = 100_000,
    rng_seed: int = 0,
) -> dict[str, float]:
    """Compute overlap RMSE between two strips using nearest-neighbour distances.

    Parameters
    ----------
    reference:
        (N, 3) reference strip.
    target:
        (M, 3) target strip.
    max_dist:
        Maximum distance (same unit as coordinates) to consider a neighbour
        'in overlap'. Points with no neighbour within this radius are excluded.
    subsample:
        If target has more than this many points, subsample randomly.
    rng_seed:
        Random seed for reproducible subsampling.

    Returns
    -------
    dict with keys: rmse, mae, n_pairs, coverage_fraction
    """
    if len(target) > subsample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(target), size=subsample, replace=False)
        query = target[idx]
    else:
        query = target

    tree = cKDTree(reference)
    dists, _ = tree.query(query, k=1, workers=-1)
    mask = dists <= max_dist
    valid = dists[mask]
    if len(valid) == 0:
        return {"rmse": np.inf, "mae": np.inf, "n_pairs": 0, "coverage_fraction": 0.0}
    return {
        "rmse": float(np.sqrt(np.mean(valid**2))),
        "mae": float(np.mean(np.abs(valid))),
        "n_pairs": int(mask.sum()),
        "coverage_fraction": float(mask.mean()),
    }
