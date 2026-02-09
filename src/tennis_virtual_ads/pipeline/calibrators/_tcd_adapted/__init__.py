"""Vendored modules adapted from TennisCourtDetector.

Original repository: https://github.com/yastrebksv/TennisCourtDetector

Only court_reference and homography logic are vendored here.
The neural network model (BallTrackerNet) is imported directly from the
sibling TennisCourtDetector repo at runtime via importlib.

Adaptations made (to remove unnecessary dependencies):
- court_reference.py: removed unused ``matplotlib`` import.
- homography.py: replaced ``scipy.spatial.distance.euclidean`` with
  ``np.linalg.norm`` to remove the scipy dependency.
"""

from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.court_reference import (
    CourtReference,
)
from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.homography import (
    get_trans_matrix,
    refer_kps,
)

__all__ = ["CourtReference", "get_trans_matrix", "refer_kps"]
