"""Lightweight scene-cut detector for broadcast tennis video.

Detects camera cuts / major viewpoint changes so that downstream
temporal components (``KeypointSmoother``, ``HomographyStabilizer``,
``JitterTracker``) can be reset immediately instead of holding stale
state across the transition.

Two independent signals are combined (OR logic):

**Signal A — Frame difference**
    Downscale current + previous frame to a small grayscale thumbnail
    and compute the mean absolute pixel difference.  Normal inter-frame
    diffs on broadcast tennis are ~2-5; hard cuts spike to 30-60+.
    A fixed threshold (default 18) cleanly separates the two.

**Signal B — Projection jump**
    Project the 14 court reference points through both the current and
    previous raw homography.  If the mean displacement exceeds a
    threshold (default 40 px), it indicates the calibrator is suddenly
    seeing a different court geometry — likely a camera switch.  Only
    evaluated when both frames have a valid ``H`` with sufficient
    confidence.

A **cooldown** window (default 10 frames) suppresses false positives
from scoreboard flashes and brief exposure changes by ignoring further
cut triggers for N frames after a confirmed cut.

Design notes
------------
- Purely observational — does not modify frames or homographies.
- No heavy dependencies — uses only OpenCV and NumPy.
- The detector is stateful; call ``reset()`` when starting a new
  video segment.
"""

from __future__ import annotations

import cv2
import numpy as np

# Default thumbnail size for frame-diff computation.
_DEFAULT_DOWNSCALE_SIZE: tuple[int, int] = (160, 90)

# Minimum confidence to trust a frame's homography for Signal B.
_MIN_CONF_FOR_PROJECTION: float = 0.2


class CutDetector:
    """Detect scene cuts via frame difference and projection jump.

    Parameters
    ----------
    reference_points : np.ndarray
        Court reference keypoints as ``(N, 1, 2)`` float32 array (the
        format expected by ``cv2.perspectiveTransform``).  Used for
        Signal B (projection jump).
    frame_diff_threshold : float
        Mean absolute pixel difference (on downscaled grayscale)
        above which a cut is signalled.  Default ``18.0``.
    projection_jump_threshold : float
        Mean projected-point displacement (in pixels) above which a
        cut is signalled.  Default ``40.0``.
    cooldown_frames : int
        Number of frames to suppress cut triggers after a confirmed
        cut, preventing false positives from brief visual transients.
        Default ``10``.
    downscale_size : tuple[int, int]
        ``(width, height)`` for the downscaled grayscale thumbnail
        used by Signal A.  Default ``(160, 90)``.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        frame_diff_threshold: float = 18.0,
        projection_jump_threshold: float = 40.0,
        cooldown_frames: int = 10,
        downscale_size: tuple[int, int] = _DEFAULT_DOWNSCALE_SIZE,
    ) -> None:
        self._reference_points = reference_points.copy().astype(np.float32)
        self._frame_diff_threshold = frame_diff_threshold
        self._projection_jump_threshold = projection_jump_threshold
        self._cooldown_frames = cooldown_frames
        self._downscale_size = downscale_size

        # --- Internal state ------------------------------------------------
        self._previous_gray_small: np.ndarray | None = None
        self._previous_homography: np.ndarray | None = None
        self._previous_confidence: float = 0.0

        self._cooldown_counter: int = 0
        self._total_cuts_detected: int = 0

        # Expose last-computed signal values for HUD / debug.
        self._last_frame_diff: float | None = None
        self._last_projection_jump: float | None = None
        self._is_cut_this_frame: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        homography_raw: np.ndarray | None,
        confidence: float,
        reproj_error: float | None,
    ) -> bool:
        """Evaluate whether the current frame is a scene cut.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image.
        homography_raw : np.ndarray | None
            Raw 3x3 court-to-image homography, or ``None`` when
            calibration failed.
        confidence : float
            Calibration confidence for this frame.
        reproj_error : float | None
            Reprojection error (currently unused; reserved).

        Returns
        -------
        bool
            ``True`` if a scene cut was detected on this frame.
        """
        self._is_cut_this_frame = False
        self._last_frame_diff = None
        self._last_projection_jump = None

        # --- Downscale current frame to grayscale thumbnail ---------------
        current_gray_small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            self._downscale_size,
            interpolation=cv2.INTER_AREA,
        )

        # --- Signal A: frame difference -----------------------------------
        signal_a_triggered = False
        if self._previous_gray_small is not None:
            frame_diff = float(
                np.mean(
                    np.abs(
                        current_gray_small.astype(np.float32)
                        - self._previous_gray_small.astype(np.float32)
                    )
                )
            )
            self._last_frame_diff = frame_diff
            signal_a_triggered = frame_diff > self._frame_diff_threshold

        # --- Signal B: projection jump ------------------------------------
        signal_b_triggered = False
        if (
            homography_raw is not None
            and confidence >= _MIN_CONF_FOR_PROJECTION
            and self._previous_homography is not None
            and self._previous_confidence >= _MIN_CONF_FOR_PROJECTION
        ):
            projected_current = cv2.perspectiveTransform(self._reference_points, homography_raw)
            projected_previous = cv2.perspectiveTransform(
                self._reference_points, self._previous_homography
            )
            per_point_displacement = np.linalg.norm(
                projected_current.reshape(-1, 2) - projected_previous.reshape(-1, 2),
                axis=1,
            )
            projection_jump = float(np.mean(per_point_displacement))
            self._last_projection_jump = projection_jump
            signal_b_triggered = projection_jump > self._projection_jump_threshold

        # --- Combine signals with cooldown --------------------------------
        is_cut = False

        if self._cooldown_counter > 0:
            # Inside cooldown window — suppress any triggers.
            self._cooldown_counter -= 1
        elif signal_a_triggered or signal_b_triggered:
            is_cut = True
            self._is_cut_this_frame = True
            self._total_cuts_detected += 1
            self._cooldown_counter = self._cooldown_frames

        # --- Update state for next frame ----------------------------------
        self._previous_gray_small = current_gray_small
        self._previous_homography = homography_raw.copy() if homography_raw is not None else None
        self._previous_confidence = confidence

        return is_cut

    def reset(self) -> None:
        """Clear all detector state.

        Call this when starting a new video segment.  Does NOT reset
        the total cuts counter (that tracks lifetime stats).
        """
        self._previous_gray_small = None
        self._previous_homography = None
        self._previous_confidence = 0.0
        self._cooldown_counter = 0
        self._last_frame_diff = None
        self._last_projection_jump = None
        self._is_cut_this_frame = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_cut_this_frame(self) -> bool:
        """Whether the last :meth:`update` call detected a cut."""
        return self._is_cut_this_frame

    @property
    def last_frame_diff(self) -> float | None:
        """Most recent frame-diff value (Signal A), or ``None``."""
        return self._last_frame_diff

    @property
    def last_projection_jump(self) -> float | None:
        """Most recent projection-jump value (Signal B), or ``None``."""
        return self._last_projection_jump

    @property
    def total_cuts_detected(self) -> int:
        """Cumulative number of cuts detected over the detector's lifetime."""
        return self._total_cuts_detected
