"""Temporal homography stabilizer using EMA in H-space.

Sits *after* the keypoint smoother (if enabled) and provides three
capabilities that keypoint-level smoothing alone cannot:

1. **EMA in H-space** — smooths the homography matrix directly, catching
   residual jitter from the best-of-12 config selector switching between
   configurations.
2. **Config-switch guard** — detects sudden projected-point jumps (even
   when reprojection error is similar) and treats them as outliers,
   preventing the overlay/ad from snapping to a new config.
3. **Hold-last-good** — when calibration fails for a few frames (e.g. a
   player obscures keypoints), the stabilizer returns the last known-good
   H instead of dropping the overlay/ad entirely.

Design notes
------------
- Homographies are defined up to scale, so both ``H_new`` and ``H_ema``
  are normalised to ``H[2,2] = 1`` every update.  This ensures that
  element-wise EMA blending produces a geometrically meaningful average.
- EMA formula: ``H_ema = alpha * H_ema + (1 - alpha) * H_new``, where
  *alpha* weights the history and ``(1 - alpha)`` weights the new
  observation.  Higher alpha → smoother / slower to react.
- The position-jump guard projects a set of reference court points
  through both ``H_new`` and ``H_ema`` and measures mean displacement.
  If the displacement exceeds ``spike_factor * median(recent)``, the
  frame is rejected.  This catches best-of-12 config switches cleanly.
- The stabilizer is stateful and should be ``reset()`` on scene cuts or
  when starting a new video segment.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class HomographyStabilizer:
    """EMA-based temporal stabilizer for court-to-image homographies.

    Parameters
    ----------
    reference_points : np.ndarray
        Court reference keypoints as ``(N, 1, 2)`` float32 array (the
        format expected by ``cv2.perspectiveTransform``).  Used for the
        config-switch position-jump guard.
    alpha : float
        EMA blending factor in ``(0, 1]``.  Higher values weight the
        *history* more heavily (smoother, slower to react).  ``1.0``
        keeps only the previous value.  Default ``0.9``.
    max_hold_frames : int
        Maximum number of consecutive frames to return the last-good
        ``H_ema`` when no valid input is received.  After this many
        frames the stabilizer gives up and returns ``None``.
        Default ``15``.
    spike_factor : float
        Outlier rejection threshold.  A frame is rejected when its
        reprojection error *or* its projected-point displacement
        exceeds ``spike_factor * median(recent_values)``.
        Default ``2.0``.
    spike_window_size : int
        Number of recent values to keep for the rolling median used
        by the outlier detectors.  Default ``30``.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        alpha: float = 0.9,
        max_hold_frames: int = 15,
        spike_factor: float = 2.0,
        spike_window_size: int = 30,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if max_hold_frames < 0:
            raise ValueError(f"max_hold_frames must be >= 0, got {max_hold_frames}")

        self._reference_points = reference_points.copy().astype(np.float32)
        self._alpha = alpha
        self._max_hold_frames = max_hold_frames
        self._spike_factor = spike_factor

        # --- Internal state ------------------------------------------------
        self._H_ema: np.ndarray | None = None
        self._hold_count: int = 0
        self._did_reject_this_frame: bool = False

        # Rolling windows for spike detection.
        self._error_history: deque[float] = deque(maxlen=spike_window_size)
        self._displacement_history: deque[float] = deque(maxlen=spike_window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        homography_raw: np.ndarray | None,
        confidence: float,
        reproj_error: float | None,
    ) -> np.ndarray | None:
        """Feed one frame's homography and return the stabilised result.

        Parameters
        ----------
        homography_raw : np.ndarray | None
            3x3 court-to-image homography, or ``None`` when calibration
            failed or was rejected by the caller.
        confidence : float
            Calibration confidence for this frame (informational; the
            caller is responsible for thresholding).
        reproj_error : float | None
            Reprojection error for this frame's homography.  ``None``
            skips the error-spike check.

        Returns
        -------
        np.ndarray | None
            Stabilised 3x3 homography, or ``None`` if no valid estimate
            is available (never had one, or hold window exhausted).
        """
        self._did_reject_this_frame = False

        # --- No valid input: hold or give up ------------------------------
        if homography_raw is None:
            return self._hold_or_none()

        # --- Normalise H_new so H[2,2] = 1 -------------------------------
        if abs(homography_raw[2, 2]) < 1e-12:
            # Degenerate matrix; treat as failure.
            return self._hold_or_none()
        homography_new = homography_raw / homography_raw[2, 2]

        # --- Outlier rejection: reprojection error spike ------------------
        if reproj_error is not None and self._is_error_spike(reproj_error):
            self._did_reject_this_frame = True
            return self._hold_or_none()

        # --- Outlier rejection: config-switch position jump ---------------
        if self._H_ema is not None and self._is_position_jump(homography_new):
            self._did_reject_this_frame = True
            return self._hold_or_none()

        # --- Accept: record into histories (only non-rejected values) -----
        if reproj_error is not None:
            self._error_history.append(reproj_error)
        if self._H_ema is not None:
            displacement = self._compute_displacement(homography_new)
            self._displacement_history.append(displacement)

        # --- EMA blend ----------------------------------------------------
        if self._H_ema is None:
            # First valid frame: initialise directly.
            self._H_ema = homography_new.copy()
        else:
            self._H_ema = self._alpha * self._H_ema + (1.0 - self._alpha) * homography_new
            # Re-normalise after blending.
            self._H_ema = self._H_ema / self._H_ema[2, 2]

        # Reset hold counter on successful update.
        self._hold_count = 0

        # self._H_ema is guaranteed non-None here (just assigned above).
        assert self._H_ema is not None
        return np.array(self._H_ema, copy=True)

    def reset(self) -> None:
        """Clear all stabilizer state.

        The next call to :meth:`update` will initialise from scratch.
        """
        self._H_ema = None
        self._hold_count = 0
        self._did_reject_this_frame = False
        self._error_history.clear()
        self._displacement_history.clear()

    @property
    def is_holding(self) -> bool:
        """``True`` when the stabilizer is returning a held last-good H."""
        return self._hold_count > 0

    @property
    def hold_count(self) -> int:
        """Number of consecutive frames the stabilizer has been holding."""
        return self._hold_count

    @property
    def did_reject_this_frame(self) -> bool:
        """``True`` if the last :meth:`update` rejected the input H."""
        return self._did_reject_this_frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hold_or_none(self) -> np.ndarray | None:
        """Return the last-good ``H_ema`` if within hold window, else ``None``."""
        if self._H_ema is None:
            return None
        self._hold_count += 1
        if self._hold_count > self._max_hold_frames:
            return None
        return np.array(self._H_ema, copy=True)

    def _is_error_spike(self, current_error: float) -> bool:
        """Return ``True`` if *current_error* is a spike relative to history."""
        if len(self._error_history) < 5:
            return False
        median_error = float(np.median(list(self._error_history)))
        if median_error < 1e-6:
            return False
        return current_error > self._spike_factor * median_error

    def _compute_displacement(self, homography_new: np.ndarray) -> float:
        """Mean pixel displacement between projections of H_new and H_ema."""
        # _compute_displacement is only called when self._H_ema is not None.
        assert self._H_ema is not None
        projected_new = cv2.perspectiveTransform(self._reference_points, homography_new)
        projected_ema = cv2.perspectiveTransform(self._reference_points, self._H_ema)
        per_point_distance = np.linalg.norm(
            projected_new.reshape(-1, 2) - projected_ema.reshape(-1, 2), axis=1
        )
        return float(np.mean(per_point_distance))

    def _is_position_jump(self, homography_new: np.ndarray) -> bool:
        """Return ``True`` if projected-point displacement is anomalously large.

        This catches best-of-12 config switches where reprojection error
        is similar but the overlay snaps to a different geometry.
        """
        displacement = self._compute_displacement(homography_new)

        if len(self._displacement_history) < 5:
            # Not enough history to judge; accept.
            return False
        median_displacement = float(np.median(list(self._displacement_history)))
        if median_displacement < 1e-6:
            return False
        return displacement > self._spike_factor * median_displacement
