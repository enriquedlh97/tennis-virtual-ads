"""Homography locker -- zero-jitter lock for static camera shots.

When the camera is static, tiny keypoint re-detection noise causes visible
jitter even after smoothing.  Professional broadcast systems solve this by
*locking* the homography: once the projected court stops moving, freeze H
and emit the same matrix every frame until the camera moves again.

State machine with hysteresis::

    UNLOCKED → (displacement < lock_threshold for lock_patience frames) → LOCKED
    LOCKED   → (displacement > unlock_threshold)                        → UNLOCKED
    LOCKED   → (explicit reset, e.g. scene cut)                         → UNLOCKED

Displacement is measured as the mean L2 pixel distance between court
reference points projected through ``H_current`` and ``H_locked``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HomographyLocker:
    """Lock the homography when the camera is static.

    Parameters
    ----------
    reference_points : np.ndarray
        Court reference keypoints as ``(N, 1, 2)`` float32 array (the
        format expected by :func:`cv2.perspectiveTransform`).
    lock_threshold : float
        Mean displacement (px) below which a frame counts as "static".
    unlock_threshold : float
        Mean displacement (px) above which a locked H is released.
    lock_patience : int
        Consecutive sub-threshold frames required before locking.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        lock_threshold: float = 0.5,
        unlock_threshold: float = 2.0,
        lock_patience: int = 5,
    ) -> None:
        self._reference_points = reference_points.astype(np.float32)
        self._lock_threshold = lock_threshold
        self._unlock_threshold = unlock_threshold
        self._lock_patience = lock_patience

        self._is_locked: bool = False
        self._locked_H: np.ndarray | None = None
        self._consecutive_static: int = 0
        self._frames_locked: int = 0
        self._last_displacement: float = 0.0

    # --- Public properties ---------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """Whether the homography is currently locked."""
        return self._is_locked

    @property
    def frames_locked(self) -> int:
        """Number of consecutive frames the lock has been held."""
        return self._frames_locked

    @property
    def last_displacement(self) -> float:
        """Mean L2 displacement (px) from the last :meth:`update` call."""
        return self._last_displacement

    # --- Core logic ----------------------------------------------------------

    def _mean_displacement(self, H_a: np.ndarray, H_b: np.ndarray) -> float:
        """Mean L2 displacement between two homographies on reference points."""
        proj_a = cv2.perspectiveTransform(self._reference_points, H_a)
        proj_b = cv2.perspectiveTransform(self._reference_points, H_b)
        return float(np.mean(np.linalg.norm(proj_a - proj_b, axis=2)))

    def update(self, homography: np.ndarray | None) -> np.ndarray | None:
        """Process one frame's homography.

        Parameters
        ----------
        homography : np.ndarray | None
            The smoothed/stabilised homography for this frame, or ``None``
            if calibration failed.

        Returns
        -------
        np.ndarray | None
            The (possibly locked) homography to use for rendering.
        """
        if homography is None:
            if self._is_locked:
                # Camera didn't move (calibration dropout) -- return locked H.
                self._frames_locked += 1
                return self._locked_H
            return None

        if self._locked_H is None:
            # First usable H -- store as reference, pass through.
            self._locked_H = homography.copy()
            self._last_displacement = 0.0
            return homography

        displacement = self._mean_displacement(homography, self._locked_H)
        self._last_displacement = displacement

        if self._is_locked:
            # --- LOCKED state ---
            if displacement > self._unlock_threshold:
                # Camera started moving -- unlock.
                self._is_locked = False
                self._locked_H = homography.copy()
                self._consecutive_static = 0
                self._frames_locked = 0
                logger.debug(
                    "H-locker UNLOCKED: displacement=%.2fpx > threshold=%.2fpx",
                    displacement,
                    self._unlock_threshold,
                )
                return homography
            else:
                # Still static -- return locked H.
                self._frames_locked += 1
                return self._locked_H
        else:
            # --- UNLOCKED state ---
            if displacement < self._lock_threshold:
                self._consecutive_static += 1
                if self._consecutive_static >= self._lock_patience:
                    # Lock!
                    self._is_locked = True
                    self._frames_locked = 1
                    logger.debug(
                        "H-locker LOCKED: %d consecutive frames < %.2fpx",
                        self._consecutive_static,
                        self._lock_threshold,
                    )
                    return self._locked_H
            else:
                self._consecutive_static = 0

            # Track latest H as the reference while unlocked.
            self._locked_H = homography.copy()
            return homography

    def reset(self) -> None:
        """Reset all state (call on scene cuts)."""
        self._is_locked = False
        self._locked_H = None
        self._consecutive_static = 0
        self._frames_locked = 0
        self._last_displacement = 0.0
