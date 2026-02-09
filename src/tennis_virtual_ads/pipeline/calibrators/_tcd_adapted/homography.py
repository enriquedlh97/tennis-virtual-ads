"""Homography computation — adapted from TennisCourtDetector.

Original source: https://github.com/yastrebksv/TennisCourtDetector
File: homography.py

Adaptations:
- Replaced ``from court_reference import CourtReference`` with a relative
  import from our vendored copy.
- Replaced ``scipy.spatial.distance.euclidean`` with ``np.linalg.norm``
  to remove the scipy dependency entirely.
- Changed ``np.Inf`` (deprecated) to ``np.inf``.
- No functional changes to the algorithm.

Algorithm overview:
    Tries all 12 court configurations (each defined by 4 reference points).
    For each configuration where the 4 corresponding keypoints were detected,
    it computes a homography (court-ref to image) and measures the mean
    reprojection error on the remaining detected keypoints.  The homography
    with the lowest mean error wins.
"""

from __future__ import annotations

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.court_reference import (
    CourtReference,
)

court_ref = CourtReference()

# All 14 reference keypoints as an (N, 1, 2) array for cv2.perspectiveTransform.
refer_kps: np.ndarray = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

# Pre-compute which keypoint indices each configuration uses.
court_conf_ind: dict[int, list[int]] = {}
for _config_index in range(len(court_ref.court_conf)):
    _conf = court_ref.court_conf[_config_index + 1]
    _indices: list[int] = []
    for _j in range(4):
        _indices.append(court_ref.key_points.index(_conf[_j]))
    court_conf_ind[_config_index + 1] = _indices


def get_trans_matrix(
    points: list[tuple[float | None, float | None]],
) -> np.ndarray | None:
    """Find the best court-to-image homography from detected keypoints.

    Parameters
    ----------
    points : list of (x, y) or (None, None)
        14 detected keypoints in image coordinates.  ``(None, None)``
        means the keypoint was not detected.

    Returns
    -------
    np.ndarray | None
        3x3 homography matrix (court-ref to image), or ``None`` if no
        valid configuration was found.
    """
    best_matrix: np.ndarray | None = None
    best_error: float = np.inf

    for config_index in range(1, 13):
        config_points = court_ref.court_conf[config_index]
        keypoint_indices = court_conf_ind[config_index]

        # The 4 detected keypoints that correspond to this configuration.
        correspondences = [
            points[keypoint_indices[0]],
            points[keypoint_indices[1]],
            points[keypoint_indices[2]],
            points[keypoint_indices[3]],
        ]

        # Skip if any of the 4 required keypoints were not detected.
        if any(None in correspondence for correspondence in correspondences):
            continue

        # Compute homography: court-ref points -> detected image points.
        source = np.array(config_points, dtype=np.float32)
        destination = np.array(correspondences, dtype=np.float32)
        matrix, _ = cv2.findHomography(source, destination, method=0)
        if matrix is None:
            continue

        # Project all reference keypoints through this homography.
        projected_keypoints = cv2.perspectiveTransform(refer_kps, matrix)

        # Measure reprojection error on keypoints NOT used to compute H.
        distances: list[float] = []
        for keypoint_index in range(12):
            if keypoint_index not in keypoint_indices and points[keypoint_index][0] is not None:
                detected = np.array(points[keypoint_index])
                projected = projected_keypoints[keypoint_index].flatten()
                distances.append(float(np.linalg.norm(detected - projected)))

        # Original code uses np.mean despite the variable name "dist_median".
        mean_error = float(np.mean(distances)) if distances else np.inf

        if mean_error < best_error:
            best_matrix = matrix
            best_error = mean_error

    return best_matrix
