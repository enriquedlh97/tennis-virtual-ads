"""Quantitative overlay-jitter measurement via projected-point acceleration.

The core idea is simple:

1. Pick a set of fixed reference points in court space (the 14 keypoints).
2. Each frame, project them through the homography ``H`` into image pixel
   coordinates.
3. Compute **acceleration** (the second derivative of position):
   ``a(t) = pos(t) - 2*pos(t-1) + pos(t-2)``.
4. Report summary statistics on the magnitude of acceleration across all
   reference points and all frames.

*Why acceleration instead of raw displacement?*

Raw frame-to-frame displacement includes both legitimate camera motion
(panning, zooming) and estimation noise (jitter).  Smooth camera motion
produces near-zero acceleration — only the "jerky" noise remains.  This
makes acceleration a robust jitter metric even when the camera is moving.

The tracker is purely observational — it does not modify any frames,
keypoints, or homographies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class JitterSummary:
    """Summary statistics for overlay jitter."""

    mean_acceleration_px: float
    p95_acceleration_px: float
    max_acceleration_px: float
    num_frames_measured: int

    def to_log_string(self) -> str:
        """Format as a single-line log message."""
        return (
            f"mean_accel={self.mean_acceleration_px:.2f}px  "
            f"p95_accel={self.p95_acceleration_px:.2f}px  "
            f"max_accel={self.max_acceleration_px:.2f}px  "
            f"({self.num_frames_measured} frames)"
        )


@dataclass
class JitterTracker:
    """Track projected-point acceleration to quantify overlay jitter.

    Usage::

        tracker = JitterTracker(reference_points=refer_kps)
        for frame_index, frame in video:
            result = calibrator.estimate(frame)
            if result["H"] is not None:
                tracker.update(result["H"])
        summary = tracker.get_summary()
        print(summary.to_log_string())

    Parameters
    ----------
    reference_points : np.ndarray
        Court reference keypoints as ``(N, 1, 2)`` array (the format
        expected by ``cv2.perspectiveTransform``).  Typically the 14
        court keypoints from ``homography.refer_kps``.
    """

    reference_points: np.ndarray

    # --- Internal state (not part of public interface) --------------------
    _previous_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _previous_velocity: np.ndarray | None = field(default=None, init=False, repr=False)
    _acceleration_magnitudes: list[float] = field(default_factory=list, init=False, repr=False)

    def update(self, homography: np.ndarray) -> None:
        """Record projected positions for one frame.

        Parameters
        ----------
        homography : np.ndarray
            3x3 court-to-image homography matrix.
        """
        # Project reference points into image space.
        projected = cv2.perspectiveTransform(self.reference_points, homography)
        # Flatten to (N, 2) for easier math.
        current_positions = projected.reshape(-1, 2)

        if self._previous_positions is not None:
            current_velocity = current_positions - self._previous_positions

            if self._previous_velocity is not None:
                # acceleration = change in velocity = pos(t) - 2*pos(t-1) + pos(t-2)
                acceleration = current_velocity - self._previous_velocity
                # Per-point acceleration magnitude, then mean across all points.
                per_point_magnitude = np.linalg.norm(acceleration, axis=1)
                mean_magnitude = float(np.mean(per_point_magnitude))
                self._acceleration_magnitudes.append(mean_magnitude)

            self._previous_velocity = current_velocity

        self._previous_positions = current_positions

    def get_summary(self) -> JitterSummary | None:
        """Compute and return jitter summary statistics.

        Returns ``None`` if fewer than 3 frames have been tracked (need
        at least 3 positions to compute one acceleration sample).
        """
        if len(self._acceleration_magnitudes) < 1:
            return None

        magnitudes = np.array(self._acceleration_magnitudes)
        return JitterSummary(
            mean_acceleration_px=float(np.mean(magnitudes)),
            p95_acceleration_px=float(np.percentile(magnitudes, 95)),
            max_acceleration_px=float(np.max(magnitudes)),
            num_frames_measured=len(magnitudes),
        )

    def reset(self) -> None:
        """Clear all tracking state."""
        self._previous_positions = None
        self._previous_velocity = None
        self._acceleration_magnitudes.clear()
