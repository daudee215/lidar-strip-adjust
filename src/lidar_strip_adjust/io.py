"""LAS/LAZ I/O helpers using laspy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

try:
    import laspy
except ImportError as exc:
    raise ImportError("laspy >= 2.5 is required: pip install laspy[lazrs]") from exc


def load_strip(path: str | Path, *, scale_to_metres: bool = True) -> NDArray[np.float64]:
    """Load a LAS/LAZ file and return an (N, 3) XYZ float64 array.

    Parameters
    ----------
    path:
        Path to a .las or .laz file.
    scale_to_metres:
        If True (default) apply the LAS scale and offset factors so that
        returned coordinates are in the file's native CRS unit (metres for
        most airborne surveys).

    Returns
    -------
    NDArray[np.float64]
        (N, 3) array of XYZ coordinates.
    """
    las = laspy.read(path)
    if scale_to_metres:
        x = np.array(las.x, dtype=np.float64)
        y = np.array(las.y, dtype=np.float64)
        z = np.array(las.z, dtype=np.float64)
    else:
        x = np.array(las.X, dtype=np.float64) * las.header.scales[0] + las.header.offsets[0]
        y = np.array(las.Y, dtype=np.float64) * las.header.scales[1] + las.header.offsets[1]
        z = np.array(las.Z, dtype=np.float64) * las.header.scales[2] + las.header.offsets[2]
    return np.column_stack([x, y, z])


def save_strip(
    points: NDArray[np.float64],
    template_path: str | Path,
    output_path: str | Path,
) -> None:
    """Write corrected XYZ back into a new LAS file, preserving all other fields.

    Parameters
    ----------
    points:
        (N, 3) corrected XYZ coordinates.
    template_path:
        Original LAS file (source of all non-XYZ attributes, header, CRS).
    output_path:
        Destination path for the adjusted LAS file.
    """
    las = laspy.read(template_path)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.write(output_path)
