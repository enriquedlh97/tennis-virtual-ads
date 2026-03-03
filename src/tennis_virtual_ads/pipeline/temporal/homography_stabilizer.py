"""Temporal homography stabilizer using EMA or Kalman filter.

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

Two filtering modes are available:

**EMA mode** (default): Element-wise exponential moving average on the
normalised homography.  Simple, effective for slow camera motion.

**Kalman mode**: Decomposes the homography into camera parameters
(pan, tilt, focal length) and applies a Kalman filter with a constant-
velocity model.  Handles smooth camera pans and brief detection gaps
much better than EMA, with predictive smoothing.

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

import logging
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalman filter helper: decompose / reconstruct homography via camera params
# ---------------------------------------------------------------------------


def _decompose_homography_to_params(
    H: np.ndarray,
    image_center: tuple[float, float] = (640.0, 360.0),
) -> np.ndarray | None:
    """Decompose a court-to-image homography into approximate camera params.

    Uses a simplified pinhole camera model to extract pan, tilt, and focal
    length from the homography.  This is an approximation that works well
    for broadcast cameras viewing a planar court.

    Parameters
    ----------
    H : np.ndarray
        3x3 normalised (H[2,2]=1) court-to-image homography.
    image_center : tuple[float, float]
        Approximate image center (cx, cy).  Default (640, 360) for 720p.

    Returns
    -------
    np.ndarray | None
        ``[pan, tilt, focal]`` in radians / pixels, or ``None`` on failure.
    """
    # Approximate focal length from the homography.
    # For a planar homography H = K [r1 r2 t], the columns of K^-1 H
    # should be orthonormal rotation columns.  We use this to estimate f.

    # Approximate K^-1 assuming cx, cy known and no skew.
    cx, cy = image_center
    # Using ||K^-1 h1|| = ||K^-1 h2|| constraint:
    # This simplifies to f^2 = -(h1[0]*h2[0] + h1[1]*h2[1]) / (h1[2]*h2[2])
    # when cx=cy=0 in normalised coords.

    # Shift H to account for principal point.
    H_shifted = H.copy()
    H_shifted[0, :] -= cx * H_shifted[2, :]
    H_shifted[1, :] -= cy * H_shifted[2, :]

    h1s = H_shifted[:, 0]
    h2s = H_shifted[:, 1]

    denom = h1s[2] * h2s[2]
    if abs(denom) < 1e-12:
        # Fallback: estimate focal from column norms.
        norm1 = np.linalg.norm(h1s[:2])
        norm2 = np.linalg.norm(h2s[:2])
        f = (norm1 + norm2) / (2.0 * max(abs(h1s[2]) + abs(h2s[2]), 1e-6))
    else:
        f_sq = -(h1s[0] * h2s[0] + h1s[1] * h2s[1]) / denom
        if f_sq <= 0:
            # Use column norm ratio as fallback.
            norm1 = np.linalg.norm(h1s[:2])
            f = norm1 / max(abs(h1s[2]), 1e-6)
        else:
            f = np.sqrt(f_sq)

    f = max(f, 100.0)  # Clamp to reasonable minimum.

    # Build approximate K^-1.
    K_inv = np.array(
        [
            [1.0 / f, 0, -cx / f],
            [0, 1.0 / f, -cy / f],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # R columns from K^-1 H.
    r1 = K_inv @ H[:, 0]
    r2 = K_inv @ H[:, 1]

    # Normalise.
    lam = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
    if lam < 1e-12:
        return None
    r1 = r1 / lam
    r2 = r2 / lam
    r3 = np.cross(r1, r2)

    # Ensure proper rotation (orthogonalise via SVD).
    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R

    # Extract pan (rotation about Y) and tilt (rotation about X).
    # R = Ry(pan) @ Rx(tilt) decomposition:
    tilt = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    pan = np.arctan2(R[1, 0], R[0, 0])

    return np.array([pan, tilt, f], dtype=np.float64)


def _reconstruct_homography_from_params(
    params: np.ndarray,
    image_center: tuple[float, float] = (640.0, 360.0),
) -> np.ndarray:
    """Reconstruct a court-to-image homography from camera parameters.

    This is the inverse of :func:`_decompose_homography_to_params`.
    Builds K @ R[:, :2] where R = Ry(pan) @ Rx(tilt).

    Parameters
    ----------
    params : np.ndarray
        ``[pan, tilt, focal]``.
    image_center : tuple[float, float]
        Principal point (cx, cy).

    Returns
    -------
    np.ndarray
        3x3 homography (normalised so H[2,2]=1).
    """
    pan, tilt, f = params[0], params[1], params[2]
    cx, cy = image_center

    # Rotation matrices.
    cos_p, sin_p = np.cos(pan), np.sin(pan)
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)

    Ry = np.array(
        [
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p],
        ]
    )
    Rx = np.array(
        [
            [1, 0, 0],
            [0, cos_t, -sin_t],
            [0, sin_t, cos_t],
        ]
    )

    R = Ry @ Rx

    K = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ]
    )

    # Pad to 3x3: H = K @ R for a pure rotation (tripod camera).
    # For a pure rotation (tripod camera), t=0, so H = K @ [r1, r2].
    # We need a 3x3 matrix: append r3 weighted column.
    H_full: np.ndarray = K @ np.column_stack([R[:, 0], R[:, 1], R[:, 2]])

    # Normalise.
    if abs(H_full[2, 2]) > 1e-12:
        H_full = np.asarray(H_full / H_full[2, 2])

    return H_full


class HomographyStabilizer:
    """Temporal stabilizer for court-to-image homographies.

    Supports two modes: EMA (exponential moving average) and Kalman
    (constant-velocity model on decomposed camera parameters).

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
        homography when no valid input is received.  Default ``15``.
    spike_factor : float
        Outlier rejection threshold.  Default ``2.0``.
    spike_window_size : int
        Number of recent values for rolling median.  Default ``30``.
    filter_mode : str
        ``"ema"`` or ``"kalman"``.  Default ``"ema"``.
    kalman_process_noise : float
        Process noise for Kalman filter (higher = more responsive).
        Default ``1e-3``.
    kalman_measurement_noise : float
        Measurement noise for Kalman filter (higher = smoother).
        Default ``1e-1``.
    image_center : tuple[float, float] | None
        Image center for Kalman decomposition.  Auto-detected from
        first frame if ``None``.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        alpha: float = 0.9,
        max_hold_frames: int = 15,
        spike_factor: float = 2.0,
        spike_window_size: int = 30,
        filter_mode: str = "ema",
        kalman_process_noise: float = 1e-3,
        kalman_measurement_noise: float = 1e-1,
        image_center: tuple[float, float] | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if max_hold_frames < 0:
            raise ValueError(f"max_hold_frames must be >= 0, got {max_hold_frames}")
        if filter_mode not in ("ema", "kalman"):
            raise ValueError(f"filter_mode must be 'ema' or 'kalman', got '{filter_mode}'")

        self._reference_points = reference_points.copy().astype(np.float32)
        self._alpha = alpha
        self._max_hold_frames = max_hold_frames
        self._spike_factor = spike_factor
        self._filter_mode = filter_mode
        self._kalman_process_noise = kalman_process_noise
        self._kalman_measurement_noise = kalman_measurement_noise
        self._image_center = image_center or (640.0, 360.0)

        # --- Internal state (shared) ---------------------------------------
        self._H_ema: np.ndarray | None = None  # Used by both modes for output
        self._hold_count: int = 0
        self._did_reject_this_frame: bool = False

        # Rolling windows for spike detection.
        self._error_history: deque[float] = deque(maxlen=spike_window_size)
        self._displacement_history: deque[float] = deque(maxlen=spike_window_size)

        # --- Kalman filter state -------------------------------------------
        self._kalman: cv2.KalmanFilter | None = None
        self._kalman_initialised: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def filter_mode(self) -> str:
        """The active filter mode (``'ema'`` or ``'kalman'``)."""
        return self._filter_mode

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
            if (
                self._filter_mode == "kalman"
                and self._kalman_initialised
                and self._kalman is not None
            ):
                # Kalman can predict forward without measurement.
                return self._kalman_predict_only()
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

        # --- Filter -------------------------------------------------------
        if self._filter_mode == "kalman":
            H_filtered = self._kalman_update(homography_new)
        else:
            H_filtered = self._ema_update(homography_new)

        if H_filtered is not None:
            self._H_ema = H_filtered.copy()

        # Reset hold counter on successful update.
        self._hold_count = 0

        return np.array(self._H_ema, copy=True) if self._H_ema is not None else None

    def reset(self) -> None:
        """Clear all stabilizer state.

        The next call to :meth:`update` will initialise from scratch.
        """
        self._H_ema = None
        self._hold_count = 0
        self._did_reject_this_frame = False
        self._error_history.clear()
        self._displacement_history.clear()
        self._kalman = None
        self._kalman_initialised = False

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
    # EMA filtering
    # ------------------------------------------------------------------

    def _ema_update(self, homography_new: np.ndarray) -> np.ndarray:
        """Apply EMA blending and return the filtered homography."""
        if self._H_ema is None:
            return np.array(homography_new, copy=True)

        H_blended: np.ndarray = self._alpha * self._H_ema + (1.0 - self._alpha) * homography_new
        # Re-normalise after blending.
        H_blended = np.asarray(H_blended / H_blended[2, 2])
        return H_blended

    # ------------------------------------------------------------------
    # Kalman filtering
    # ------------------------------------------------------------------

    def _init_kalman(self, initial_params: np.ndarray) -> None:
        """Initialise the Kalman filter with 6-state constant-velocity model.

        State: [pan, tilt, focal, d_pan, d_tilt, d_focal]
        Measurement: [pan, tilt, focal]
        """
        kf = cv2.KalmanFilter(6, 3, 0)

        # Transition matrix: constant velocity.
        kf.transitionMatrix = np.eye(6, dtype=np.float32)
        kf.transitionMatrix[0, 3] = 1.0  # pan += d_pan
        kf.transitionMatrix[1, 4] = 1.0  # tilt += d_tilt
        kf.transitionMatrix[2, 5] = 1.0  # focal += d_focal

        # Measurement matrix: observe [pan, tilt, focal] directly.
        kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1.0
        kf.measurementMatrix[1, 1] = 1.0
        kf.measurementMatrix[2, 2] = 1.0

        # Process noise.
        q = self._kalman_process_noise
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * q
        # Higher noise for velocity states (less certainty about velocity).
        kf.processNoiseCov[3, 3] = q * 10.0
        kf.processNoiseCov[4, 4] = q * 10.0
        kf.processNoiseCov[5, 5] = q * 10.0

        # Measurement noise.
        r = self._kalman_measurement_noise
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * r

        # Initial state.
        kf.statePost = np.array(
            [initial_params[0], initial_params[1], initial_params[2], 0.0, 0.0, 0.0],
            dtype=np.float32,
        ).reshape(6, 1)

        # Initial error covariance.
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1

        self._kalman = kf
        self._kalman_initialised = True

    def _kalman_update(self, homography_new: np.ndarray) -> np.ndarray | None:
        """Decompose H, update Kalman filter, reconstruct filtered H."""
        params = _decompose_homography_to_params(homography_new, self._image_center)
        if params is None:
            # Decomposition failed — fall back to EMA for this frame.
            return self._ema_update(homography_new)

        if not self._kalman_initialised:
            self._init_kalman(params)
            return np.array(homography_new, copy=True)

        assert self._kalman is not None

        # Predict.
        self._kalman.predict()

        # Correct with measurement.
        measurement = params.astype(np.float32).reshape(3, 1)
        self._kalman.correct(measurement)

        # Extract filtered state.
        state = self._kalman.statePost.flatten()
        filtered_params = np.array([state[0], state[1], state[2]], dtype=np.float64)

        # Reconstruct homography.
        H_filtered = _reconstruct_homography_from_params(filtered_params, self._image_center)

        # Normalise.
        if abs(H_filtered[2, 2]) > 1e-12:
            H_filtered = np.asarray(H_filtered / H_filtered[2, 2])

        return H_filtered

    def _kalman_predict_only(self) -> np.ndarray | None:
        """Predict without measurement (for hold frames in Kalman mode)."""
        if self._kalman is None:
            return self._hold_or_none()

        # Use predict to advance state.
        predicted = self._kalman.predict()
        state = predicted.flatten()
        filtered_params = np.array([state[0], state[1], state[2]], dtype=np.float64)

        H_predicted = _reconstruct_homography_from_params(filtered_params, self._image_center)
        if abs(H_predicted[2, 2]) > 1e-12:
            H_predicted = np.asarray(H_predicted / H_predicted[2, 2])

        # Still count as holding for the hold limit.
        self._hold_count += 1
        if self._hold_count > self._max_hold_frames:
            return None

        self._H_ema = H_predicted.copy()
        return np.array(H_predicted, copy=True)

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
