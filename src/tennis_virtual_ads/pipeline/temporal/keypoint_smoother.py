"""Temporal keypoint smoother using exponential moving average (EMA).

Reduces frame-to-frame jitter in detected court keypoints by blending
each new observation with the running smoothed estimate.  A smoothed
keypoint array can then be fed into the homography selector to produce
a more stable court-to-image homography.

Design notes
------------
- Smoothing is **per-keypoint**: each of the 14 keypoints has its own
  independent EMA state.  If a keypoint is undetected (NaN) in a frame,
  the smoothed value is carried forward unchanged.
- An **error spike detector** monitors the reprojection error over a
  rolling window.  If the current error exceeds ``spike_factor`` times
  the running median, the smoother resets to prevent drift accumulation
  (e.g. after a scene cut or a very bad detection).
- The smoother is stateful and must be ``reset()`` between independent
  video segments.
"""

from __future__ import annotations

from collections import deque
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

NUM_KEYPOINTS = 14
KeypointsArray: TypeAlias = NDArray[np.float32]


class KeypointSmoother:
    """EMA-based temporal smoother for court keypoints.

    Parameters
    ----------
    alpha : float
        EMA blending factor in ``(0, 1]``.  Higher values weight the new
        observation more heavily (more responsive, less smooth).
        ``alpha=1.0`` disables smoothing entirely.  Default ``0.7``.
    enable_spike_reset : bool
        If ``True``, automatically reset the smoother when the
        reprojection error spikes.  Default ``True``.
    spike_factor : float
        Reset threshold: if ``current_error > spike_factor * median(recent_errors)``,
        trigger a reset.  Default ``2.0``.
    spike_window_size : int
        Number of recent error values to keep for the rolling median.
        Default ``30``.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        enable_spike_reset: bool = True,
        spike_factor: float = 2.0,
        spike_window_size: int = 30,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self._alpha = alpha
        self._enable_spike_reset = enable_spike_reset
        self._spike_factor = spike_factor

        self._smoothed: KeypointsArray | None = None
        self._error_history: deque[float] = deque(maxlen=spike_window_size)
        self._did_reset_this_frame = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        raw_keypoints: NDArray[np.floating[Any]],
        reprojection_error: float | None = None,
    ) -> KeypointsArray:
        """Feed a new frame's keypoints and return the smoothed result.

        Parameters
        ----------
        raw_keypoints : np.ndarray
            ``(14, 2)`` array of detected keypoints.  Undetected points
            should be ``(NaN, NaN)``.
        reprojection_error : float | None
            Reprojection error for this frame's raw homography.  Used for
            spike detection.  ``None`` skips the spike check.

        Returns
        -------
        np.ndarray
            ``(14, 2)`` smoothed keypoints.
        """
        self._did_reset_this_frame = False
        raw_keypoints_float32 = cast(KeypointsArray, raw_keypoints.astype(np.float32, copy=False))

        # --- Spike detection (before smoothing) ---------------------------
        if (
            self._enable_spike_reset
            and reprojection_error is not None
            and self._should_reset(reprojection_error)
        ):
            self.reset()
            self._did_reset_this_frame = True

        # Track error history (after potential reset, so the spike value
        # doesn't poison the window).
        if reprojection_error is not None and not self._did_reset_this_frame:
            self._error_history.append(reprojection_error)

        # --- EMA smoothing ------------------------------------------------
        if self._smoothed is None:
            # First frame: initialise directly from raw observations.
            self._smoothed = cast(KeypointsArray, raw_keypoints_float32.copy())
            return cast(KeypointsArray, self._smoothed.copy())

        smoothed = self._smoothed
        alpha = self._alpha

        for keypoint_index in range(NUM_KEYPOINTS):
            raw_x, raw_y = raw_keypoints_float32[keypoint_index]
            raw_is_valid = not (np.isnan(raw_x) or np.isnan(raw_y))
            prev_is_valid = not (
                np.isnan(smoothed[keypoint_index, 0]) or np.isnan(smoothed[keypoint_index, 1])
            )

            if raw_is_valid and prev_is_valid:
                # Normal case: blend new observation with previous.
                smoothed[keypoint_index, 0] = (
                    alpha * raw_x + (1 - alpha) * smoothed[keypoint_index, 0]
                )
                smoothed[keypoint_index, 1] = (
                    alpha * raw_y + (1 - alpha) * smoothed[keypoint_index, 1]
                )
            elif raw_is_valid and not prev_is_valid:
                # First detection of this keypoint: initialise directly.
                smoothed[keypoint_index] = [raw_x, raw_y]
            # else: raw is NaN -- carry forward previous smoothed value
            # (or NaN if never detected).

        self._smoothed = smoothed
        return cast(KeypointsArray, smoothed.copy())

    def reset(self) -> None:
        """Clear all smoothing state.

        The next call to :meth:`update` will initialise from scratch.
        """
        self._smoothed = None
        self._error_history.clear()

    @property
    def did_reset_this_frame(self) -> bool:
        """Whether the smoother auto-reset during the last :meth:`update` call."""
        return self._did_reset_this_frame

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_reset(self, current_error: float) -> bool:
        """Return ``True`` if *current_error* is a spike relative to history."""
        if len(self._error_history) < 5:
            # Not enough history to judge; don't reset.
            return False
        median_error = float(np.median(list(self._error_history)))
        if median_error < 1e-6:
            return False
        return current_error > self._spike_factor * median_error
