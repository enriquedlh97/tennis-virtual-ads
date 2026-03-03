"""Lucas-Kanade optical flow keypoint tracker.

Wraps the court calibrator and tracks detected keypoints between full
detections using ``cv2.calcOpticalFlowPyrLK()``.  This eliminates the
frame-to-frame jitter caused by running BallTrackerNet independently on
every frame.

Design overview (inspired by BroadTrack, WACV 2025):
    1. **Keyframes**: Run full calibrator detection every N frames, or when
       confidence drops below a threshold, or on scene cuts.
    2. **Tracking frames**: Track keypoints from the previous frame using
       Lucas-Kanade pyramid optical flow with forward-backward verification.
    3. **Redetection triggers**: Drop to full detection when tracked points
       fall below a minimum count, reprojection error spikes, or every N
       frames as a safety net.

BroadTrack uses LK window 55x55 with 2 pyramid levels and forward-backward
verification for robustness.  We adopt similar parameters.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.calibrators.base import CalibrationResult, CourtCalibrator

logger = logging.getLogger(__name__)


class KeypointTracker:
    """Track court keypoints between full calibrator detections using LK flow.

    Parameters
    ----------
    calibrator : CourtCalibrator
        The underlying calibrator to run on keyframes.
    redetect_interval : int
        Maximum frames between full detections (safety net).  Default ``12``.
    min_tracked_points : int
        Minimum tracked keypoints before forcing a re-detection.  Default ``6``.
    fb_error_threshold : float
        Forward-backward error threshold (pixels).  Points with round-trip
        error exceeding this are dropped.  Default ``1.0``.
    lk_win_size : int
        Lucas-Kanade search window size.  Default ``55`` (matching BroadTrack).
    lk_max_level : int
        Pyramid levels for LK.  Default ``2``.
    reproj_error_threshold : float
        Maximum reprojection error before forcing re-detection.  Default ``15.0``.
    """

    def __init__(
        self,
        calibrator: CourtCalibrator,
        redetect_interval: int = 12,
        min_tracked_points: int = 6,
        fb_error_threshold: float = 1.0,
        lk_win_size: int = 55,
        lk_max_level: int = 2,
        reproj_error_threshold: float = 15.0,
    ) -> None:
        self._calibrator = calibrator
        self._redetect_interval = redetect_interval
        self._min_tracked_points = min_tracked_points
        self._fb_error_threshold = fb_error_threshold
        self._lk_win_size = lk_win_size
        self._lk_max_level = lk_max_level
        self._reproj_error_threshold = reproj_error_threshold

        # LK parameters (matching BroadTrack: 10 iters, eps=0.03).
        self._lk_win_size_tuple: tuple[int, int] = (lk_win_size, lk_win_size)
        self._lk_criteria: tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        )

        # --- Internal state ---
        self._prev_gray: np.ndarray | None = None
        self._tracked_points: np.ndarray | None = None  # (N, 2) float32
        self._tracked_indices: list[int] | None = None  # Which of the 14 kps
        self._frames_since_detection: int = 0
        self._last_detection_result: CalibrationResult | None = None

    def update(
        self,
        frame: np.ndarray,
        force_detect: bool = False,
    ) -> CalibrationResult:
        """Process one frame: detect or track keypoints.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image.
        force_detect : bool
            Force full calibrator detection (e.g. on scene cut).

        Returns
        -------
        CalibrationResult
            Contains keypoints (tracked or detected), homography, confidence.
        """
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        need_detect = (
            force_detect
            or self._prev_gray is None
            or self._tracked_points is None
            or self._tracked_indices is None
            or len(self._tracked_points) < self._min_tracked_points
            or self._frames_since_detection >= self._redetect_interval
        )

        if need_detect:
            result = self._run_detection(frame, current_gray)
        else:
            result = self._run_tracking(frame, current_gray)

        self._prev_gray = current_gray
        return result

    def reset(self) -> None:
        """Clear all tracker state (call on scene cuts)."""
        self._prev_gray = None
        self._tracked_points = None
        self._tracked_indices = None
        self._frames_since_detection = 0
        self._last_detection_result = None

    @property
    def frames_since_detection(self) -> int:
        """Number of frames since last full calibrator detection."""
        return self._frames_since_detection

    @property
    def tracked_point_count(self) -> int:
        """Number of currently tracked keypoints."""
        if self._tracked_points is None:
            return 0
        return len(self._tracked_points)

    @property
    def is_tracking(self) -> bool:
        """True when using optical flow tracking (vs full detection)."""
        return self._frames_since_detection > 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_detection(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> CalibrationResult:
        """Run the full calibrator and initialise tracking state."""
        result = self._calibrator.estimate(frame)
        self._frames_since_detection = 0
        self._last_detection_result = result

        keypoints = result["keypoints"]
        if keypoints is not None and result["H"] is not None:
            # Extract detected (non-NaN) keypoints for tracking.
            tracked_pts: list[np.ndarray] = []
            tracked_idx: list[int] = []
            for i in range(keypoints.shape[0]):
                if not (np.isnan(keypoints[i, 0]) or np.isnan(keypoints[i, 1])):
                    tracked_pts.append(keypoints[i].astype(np.float32))
                    tracked_idx.append(i)

            if tracked_pts:
                self._tracked_points = np.array(tracked_pts, dtype=np.float32)
                self._tracked_indices = tracked_idx
            else:
                self._tracked_points = None
                self._tracked_indices = None
        else:
            self._tracked_points = None
            self._tracked_indices = None

        # Add tracker debug info.
        result["debug"]["tracker_method"] = "detect"
        result["debug"]["tracked_point_count"] = (
            len(self._tracked_points) if self._tracked_points is not None else 0
        )

        return result

    def _run_tracking(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
    ) -> CalibrationResult:
        """Track keypoints from previous frame using LK optical flow."""
        assert self._prev_gray is not None
        assert self._tracked_points is not None
        assert self._tracked_indices is not None

        # Reshape for calcOpticalFlowPyrLK: (N, 1, 2).
        prev_pts = self._tracked_points.reshape(-1, 1, 2)

        # Forward tracking: prev -> current.
        # Note: OpenCV accepts None for nextPts at runtime but the stubs
        # don't reflect this, hence the type: ignore on the call.
        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            self._prev_gray,
            gray,
            prev_pts,
            None,  # nextPts (let OpenCV allocate)
            None,  # status
            None,  # err
            winSize=self._lk_win_size_tuple,
            maxLevel=self._lk_max_level,
            criteria=self._lk_criteria,
        )

        if next_pts is None or status_fwd is None:
            # LK failed completely — fall back to detection.
            return self._run_detection(frame, gray)

        # Backward tracking: current -> prev (forward-backward verification).
        back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            gray,
            self._prev_gray,
            next_pts,
            None,  # nextPts
            None,  # status
            None,  # err
            winSize=self._lk_win_size_tuple,
            maxLevel=self._lk_max_level,
            criteria=self._lk_criteria,
        )

        if back_pts is None or status_bwd is None:
            return self._run_detection(frame, gray)

        # Filter: keep points where both forward and backward succeeded
        # and the round-trip error is below threshold.
        good_mask = np.ones(len(prev_pts), dtype=bool)

        # Status check.
        good_mask &= status_fwd.flatten().astype(bool)
        good_mask &= status_bwd.flatten().astype(bool)

        # Forward-backward error check.
        fb_error = np.linalg.norm(prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)
        good_mask &= fb_error < self._fb_error_threshold

        # Apply filter.
        surviving_pts = next_pts[good_mask].reshape(-1, 2)
        surviving_indices = [
            idx for idx, keep in zip(self._tracked_indices, good_mask, strict=True) if keep
        ]

        if len(surviving_pts) < self._min_tracked_points:
            # Not enough tracked points — re-detect.
            logger.debug(
                "Tracker: only %d points survived (need %d), re-detecting",
                len(surviving_pts),
                self._min_tracked_points,
            )
            return self._run_detection(frame, gray)

        # Build the 14-keypoint array (NaN for untracked).
        keypoints_14 = np.full((14, 2), np.nan, dtype=np.float32)
        for pt, idx in zip(surviving_pts, surviving_indices, strict=True):
            keypoints_14[idx] = pt

        # Compute homography from tracked points using RANSAC.
        from tennis_virtual_ads.pipeline.calibrators._tcd_adapted.homography import (
            get_trans_matrix_ext,
            refer_kps,
        )

        kps_tuples: list[tuple[float | None, float | None]] = []
        for row in keypoints_14:
            if np.isnan(row[0]) or np.isnan(row[1]):
                kps_tuples.append((None, None))
            else:
                kps_tuples.append((float(row[0]), float(row[1])))

        h_result = get_trans_matrix_ext(kps_tuples)
        homography = h_result.matrix

        # Compute reprojection error if we have a homography.
        reproj_error: float | None = None
        if homography is not None:
            projected = cv2.perspectiveTransform(refer_kps, homography)
            errors: list[float] = []
            for pt, idx in zip(surviving_pts, surviving_indices, strict=True):
                proj_pt = projected[idx].flatten()
                errors.append(float(np.linalg.norm(pt - proj_pt)))
            reproj_error = float(np.mean(errors)) if errors else None

            # Check if reproj error is too high — trigger re-detection.
            if reproj_error is not None and reproj_error > self._reproj_error_threshold:
                logger.debug(
                    "Tracker: reproj error %.1f > threshold %.1f, re-detecting",
                    reproj_error,
                    self._reproj_error_threshold,
                )
                return self._run_detection(frame, gray)

        # Compute confidence from tracked point count.
        confidence = len(surviving_pts) / 14.0

        # Update state.
        self._tracked_points = surviving_pts.astype(np.float32)
        self._tracked_indices = surviving_indices
        self._frames_since_detection += 1

        debug: dict[str, Any] = {
            "detected_keypoint_count": len(surviving_pts),
            "total_keypoints": 14,
            "reprojection_error_px": reproj_error,
            "tracker_method": "track",
            "tracked_point_count": len(surviving_pts),
            "frames_since_detection": self._frames_since_detection,
            "fb_errors": fb_error[good_mask].tolist() if np.any(good_mask) else [],
            "dropped_count": int(np.sum(~good_mask)),
            "homography_method": h_result.method,
        }

        if h_result.inlier_mask is not None:
            debug["ransac_inlier_mask"] = h_result.inlier_mask.tolist()

        return CalibrationResult(
            H=homography,
            conf=confidence,
            keypoints=keypoints_14,
            debug=debug,
        )
