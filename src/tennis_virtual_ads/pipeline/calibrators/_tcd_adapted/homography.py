"""Homography computation — adapted from TennisCourtDetector.

Original source: https://github.com/yastrebksv/TennisCourtDetector
File: homography.py

Adaptations:
- Replaced ``from court_reference import CourtReference`` with a relative
  import from our vendored copy.
- Replaced ``scipy.spatial.distance.euclidean`` with ``np.linalg.norm``
  to remove the scipy dependency entirely.
- Changed ``np.Inf`` (deprecated) to ``np.inf``.
- Added RANSAC all-keypoints mode: when >=5 detected keypoints, uses
  ``cv2.findHomography(..., cv2.RANSAC)`` on ALL detected keypoints for
  a more robust homography.  Falls back to the original best-of-12
  configuration selector when fewer than 5 keypoints are detected.

Algorithm overview (RANSAC mode):
    Collect all detected keypoints and their court-reference correspondences.
    Compute a homography using RANSAC over all pairs simultaneously, which
    is strictly more robust than the original best-of-4 approach.

Algorithm overview (fallback mode):
    Tries all 12 court configurations (each defined by 4 reference points).
    For each configuration where the 4 corresponding keypoints were detected,
    it computes a homography (court-ref to image) and measures the mean
    reprojection error on the remaining detected keypoints.  The homography
    with the lowest mean error wins.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.court_reference import (
    CourtReference,
)

logger = logging.getLogger(__name__)

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

# Minimum detected keypoints to use RANSAC all-keypoints mode.
_MIN_KEYPOINTS_FOR_RANSAC: int = 5


class HomographyResult(NamedTuple):
    """Result of homography computation with debug info."""

    matrix: np.ndarray | None
    inlier_mask: np.ndarray | None  # Boolean mask over detected keypoints (RANSAC mode only)
    method: str  # "ransac" or "best_of_12"


def _get_trans_matrix_ransac(
    points: list[tuple[float | None, float | None]],
    ransac_reproj_threshold: float = 5.0,
) -> HomographyResult:
    """Compute homography using RANSAC over all detected keypoints.

    Parameters
    ----------
    points : list of (x, y) or (None, None)
        14 detected keypoints in image coordinates.
    ransac_reproj_threshold : float
        Maximum reprojection error (pixels) for a point to be considered
        an inlier by RANSAC.  Default ``5.0``.

    Returns
    -------
    HomographyResult
        Homography matrix, inlier mask, and method string.
    """
    # Collect all detected keypoints and their court-reference correspondences.
    src_points: list[tuple[float, float]] = []
    dst_points: list[tuple[float, float]] = []
    detected_indices: list[int] = []

    for idx in range(len(court_ref.key_points)):
        if points[idx][0] is not None and points[idx][1] is not None:
            src_points.append(court_ref.key_points[idx])
            dst_points.append((points[idx][0], points[idx][1]))  # type: ignore[arg-type]
            detected_indices.append(idx)

    if len(src_points) < _MIN_KEYPOINTS_FOR_RANSAC:
        return HomographyResult(matrix=None, inlier_mask=None, method="ransac")

    src_array = np.array(src_points, dtype=np.float32)
    dst_array = np.array(dst_points, dtype=np.float32)

    matrix, mask = cv2.findHomography(src_array, dst_array, cv2.RANSAC, ransac_reproj_threshold)

    if matrix is None:
        return HomographyResult(matrix=None, inlier_mask=None, method="ransac")

    # Build a full 14-element inlier mask (None for undetected keypoints).
    full_inlier_mask = np.zeros(len(court_ref.key_points), dtype=bool)
    if mask is not None:
        for i, kp_idx in enumerate(detected_indices):
            full_inlier_mask[kp_idx] = bool(mask[i, 0])

    return HomographyResult(matrix=matrix, inlier_mask=full_inlier_mask, method="ransac")


def _get_trans_matrix_best_of_12(
    points: list[tuple[float | None, float | None]],
) -> HomographyResult:
    """Original best-of-12 configuration homography computation.

    Tries all 12 court configurations (each defined by 4 reference points).
    For each configuration where the 4 corresponding keypoints were detected,
    it computes a homography and picks the one with lowest reprojection error.
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

        mean_error = float(np.mean(distances)) if distances else np.inf

        if mean_error < best_error:
            best_matrix = matrix
            best_error = mean_error

    return HomographyResult(matrix=best_matrix, inlier_mask=None, method="best_of_12")


def get_trans_matrix(
    points: list[tuple[float | None, float | None]],
    ransac_reproj_threshold: float = 5.0,
) -> np.ndarray | None:
    """Find the best court-to-image homography from detected keypoints.

    Uses RANSAC over all detected keypoints when >=5 are available (more
    robust).  Falls back to the original best-of-12 configuration selector
    when fewer keypoints are detected.

    Parameters
    ----------
    points : list of (x, y) or (None, None)
        14 detected keypoints in image coordinates.  ``(None, None)``
        means the keypoint was not detected.
    ransac_reproj_threshold : float
        Maximum reprojection error for RANSAC inlier classification.
        Default ``5.0`` pixels.

    Returns
    -------
    np.ndarray | None
        3x3 homography matrix (court-ref to image), or ``None`` if no
        valid configuration was found.
    """
    result = get_trans_matrix_ext(points, ransac_reproj_threshold)
    return result.matrix


def get_trans_matrix_ext(
    points: list[tuple[float | None, float | None]],
    ransac_reproj_threshold: float = 5.0,
) -> HomographyResult:
    """Find the best court-to-image homography with extended debug info.

    Same as :func:`get_trans_matrix` but returns a :class:`HomographyResult`
    with the inlier mask and method used.
    """
    # Count detected keypoints.
    detected_count = sum(1 for p in points if p[0] is not None)

    if detected_count >= _MIN_KEYPOINTS_FOR_RANSAC:
        result = _get_trans_matrix_ransac(points, ransac_reproj_threshold)
        if result.matrix is not None:
            return result
        # RANSAC failed — fall through to best-of-12.

    return _get_trans_matrix_best_of_12(points)
