"""Command-line interface for lidar-strip-adjust."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from lidar_strip_adjust import StripAdjuster, compute_strip_rmse, load_strip, save_strip


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lidar-strip-adjust",
        description=(
            "Automated boresight calibration and strip adjustment for "
            "airborne/UAV LiDAR point clouds (LAS/LAZ)."
        ),
    )
    p.add_argument("reference", type=Path, help="Reference strip (.las/.laz)")
    p.add_argument("target", type=Path, help="Target strip to adjust (.las/.laz)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output path for adjusted strip (default: <target>_adjusted.las)")
    p.add_argument("--k", type=int, default=20, metavar="K",
                   help="k-NN neighbourhood size for PCA (default: 20)")
    p.add_argument("--planarity", type=float, default=0.7,
                   help="Minimum planarity threshold [0,1] (default: 0.7)")
    p.add_argument("--max-patches", type=int, default=50_000,
                   help="Maximum patches per strip (default: 50000)")
    p.add_argument("--max-dist", type=float, default=1.0,
                   help="Maximum patch-centroid distance for matching (default: 1.0)")
    p.add_argument("--max-angle", type=float, default=15.0,
                   help="Maximum normal-angle for matching, degrees (default: 15)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    ref_path: Path = args.reference
    tgt_path: Path = args.target

    if not ref_path.exists():
        print(f"Error: reference file not found: {ref_path}", file=sys.stderr)
        return 1
    if not tgt_path.exists():
        print(f"Error: target file not found: {tgt_path}", file=sys.stderr)
        return 1

    output: Path = args.output or tgt_path.with_stem(tgt_path.stem + "_adjusted")

    print(f"Loading reference: {ref_path}")
    reference = load_strip(ref_path)
    print(f"  {len(reference):,} points")

    print(f"Loading target: {tgt_path}")
    target = load_strip(tgt_path)
    print(f"  {len(target):,} points")

    metrics_before = compute_strip_rmse(reference, target)
    print(f"\nOverlap RMSE before: {metrics_before['rmse']:.4f} m  "
          f"(n={metrics_before['n_pairs']:,}, coverage={metrics_before['coverage_fraction']:.1%})")

    adjuster = StripAdjuster(
        k_neighbours=args.k,
        planarity_threshold=args.planarity,
        max_patches=args.max_patches,
        max_correspondence_dist=args.max_dist,
        max_angle_deg=args.max_angle,
    )

    result = adjuster.adjust(reference, target, strip_id=tgt_path.stem)

    print(f"\nBoresight correction:")
    print(f"  Δω={np.degrees(result.params[0]):+.4f}°  "
          f"Δφ={np.degrees(result.params[1]):+.4f}°  "
          f"Δκ={np.degrees(result.params[2]):+.4f}°")
    print(f"  Δx={result.params[3]:+.4f}  "
          f"Δy={result.params[4]:+.4f}  "
          f"Δz={result.params[5]:+.4f}")
    print(f"\nPoint-to-plane RMSE: {result.rmse_before:.4f} → {result.rmse_after:.4f} m")
    print(f"Correspondences: {result.n_correspondences}  Converged: {result.converged}")

    if result.corrected_points is not None:
        metrics_after = compute_strip_rmse(reference, result.corrected_points)
        print(f"Overlap RMSE  after: {metrics_after['rmse']:.4f} m")

    save_strip(result.corrected_points, tgt_path, output)
    print(f"\nAdjusted strip written to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
