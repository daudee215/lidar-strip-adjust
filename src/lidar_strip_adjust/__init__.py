"""
lidar-strip-adjust
==================
Automated boresight calibration and strip adjustment for airborne/UAV LiDAR
point clouds.

Core algorithm
--------------
1. Extract planar-neighbourhood features from each strip using PCA on local
   k-NN patches (normal space sampling).
2. Build cross-strip plane correspondences by angular and distance proximity.
3. Minimise the point-to-plane distance objective over boresight rotation
   (roll Δω, pitch Δφ, yaw Δκ) plus lever-arm translation (Δx, Δy, Δz)
   using Levenberg–Marquardt (scipy.optimize.least_squares).
4. Apply the estimated rigid correction to all points in the target strip.
5. Report per-strip RMSE, point-to-plane residuals, and convergence stats.

References
----------
- Glira et al. (2015). "Rigorous Boresight Self-Calibration of Mobile and UAV
  LiDAR Scanning Systems by Strip Adjustment." Remote Sensing 11(4):442.
  https://doi.org/10.3390/rs11040442
- Tian et al. (2022). "Automatic Calibration Method for Airborne LiDAR
  Systems Based on Approximate Corresponding Points Model."
  J. Sensors 2022:4853419. https://doi.org/10.1155/2022/4853419
- Zhang et al. (2023). "Airborne LiDAR Strip Adjustment Method Based on
  Point Clouds with Planar Neighborhoods." Remote Sensing 15(23):5447.
  https://doi.org/10.3390/rs15235447
"""

from lidar_strip_adjust.adjustment import StripAdjuster
from lidar_strip_adjust.features import extract_planar_features
from lidar_strip_adjust.io import load_strip, save_strip
from lidar_strip_adjust.metrics import compute_strip_rmse

__all__ = [
    "StripAdjuster",
    "load_strip",
    "save_strip",
    "extract_planar_features",
    "compute_strip_rmse",
]

__version__ = "0.1.0"
