"""Per-frame stability metrics for quantitative pipeline evaluation.

Collects four signals every frame:

1. **Reprojection error** -- mean L2 distance of projected court keypoints
   vs their detected positions (measures calibration accuracy).
2. **H-difference (Frobenius norm)** -- ``||H_t - H_{t-1}||_F``, measures
   total geometric change between consecutive frames.
3. **Acceleration** -- second derivative of projected reference point
   positions (reuses the JitterTracker logic).
4. **Coverage** -- counts of accepted / held / rejected / cut frames.

At the end of a run, :meth:`StabilityCollector.report` prints a human-
readable summary and optionally writes a JSON file for automated comparison.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StabilityReport:
    """Summary statistics from a :class:`StabilityCollector`."""

    # Reprojection error (pixels).
    reproj_mean: float
    reproj_p95: float
    reproj_max: float

    # H-difference Frobenius norm.
    hdiff_mean: float
    hdiff_p95: float
    hdiff_max: float

    # Jitter acceleration (pixels).
    accel_mean: float
    accel_p95: float
    accel_max: float

    # Coverage.
    total_frames: int
    accepted: int
    held: int
    rejected: int
    cuts: int

    def to_log_string(self) -> str:
        """Format as a multi-line log block."""
        accept_pct = 100.0 * self.accepted / max(self.total_frames, 1)
        return (
            "=== STABILITY REPORT ===\n"
            f"Reprojection:  mean={self.reproj_mean:.1f}px  "
            f"p95={self.reproj_p95:.1f}px  max={self.reproj_max:.1f}px\n"
            f"H-difference:  mean={self.hdiff_mean:.4f}  "
            f"p95={self.hdiff_p95:.4f}  max={self.hdiff_max:.4f}\n"
            f"Jitter(accel): mean={self.accel_mean:.1f}px  "
            f"p95={self.accel_p95:.1f}px  max={self.accel_max:.1f}px\n"
            f"Coverage:      accepted={self.accepted}/{self.total_frames} ({accept_pct:.1f}%)  "
            f"held={self.held}  rejected={self.rejected}  cuts={self.cuts}"
        )

    def to_dict(self) -> dict[str, float | int]:
        """Serialise to a JSON-friendly dict."""
        return {
            "reproj_mean": self.reproj_mean,
            "reproj_p95": self.reproj_p95,
            "reproj_max": self.reproj_max,
            "hdiff_mean": self.hdiff_mean,
            "hdiff_p95": self.hdiff_p95,
            "hdiff_max": self.hdiff_max,
            "accel_mean": self.accel_mean,
            "accel_p95": self.accel_p95,
            "accel_max": self.accel_max,
            "total_frames": self.total_frames,
            "accepted": self.accepted,
            "held": self.held,
            "rejected": self.rejected,
            "cuts": self.cuts,
        }


@dataclass
class StabilityCollector:
    """Accumulate per-frame stability signals and produce a summary report.

    Parameters
    ----------
    reference_points : np.ndarray
        Court reference keypoints as ``(N, 1, 2)`` float32 array (the
        format expected by ``cv2.perspectiveTransform``).
    """

    reference_points: np.ndarray

    # --- Internal accumulators ------------------------------------------------
    _reproj_errors: list[float] = field(default_factory=list, init=False, repr=False)
    _hdiff_norms: list[float] = field(default_factory=list, init=False, repr=False)
    _accel_magnitudes: list[float] = field(default_factory=list, init=False, repr=False)

    _prev_H: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_positions: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_velocity: np.ndarray | None = field(default=None, init=False, repr=False)

    _total_frames: int = field(default=0, init=False, repr=False)
    _accepted: int = field(default=0, init=False, repr=False)
    _held: int = field(default=0, init=False, repr=False)
    _rejected: int = field(default=0, init=False, repr=False)
    _cuts: int = field(default=0, init=False, repr=False)

    def update(
        self,
        homography: np.ndarray | None,
        *,
        reproj_error: float | None = None,
        is_accepted: bool = False,
        is_held: bool = False,
        is_cut: bool = False,
    ) -> None:
        """Record one frame's metrics.

        Parameters
        ----------
        homography : np.ndarray | None
            The final (smoothed / stabilised) homography used for rendering
            this frame, or ``None`` when no usable H was available.
        reproj_error : float | None
            Reprojection error in pixels (from calibrator or smoother).
        is_accepted : bool
            Whether the calibration was accepted this frame.
        is_held : bool
            Whether a held last-good H was used.
        is_cut : bool
            Whether a scene cut was detected.
        """
        self._total_frames += 1

        # --- Coverage counters ---
        if is_cut:
            self._cuts += 1
        if is_accepted:
            self._accepted += 1
        elif is_held:
            self._held += 1
        else:
            self._rejected += 1

        # --- Reprojection error ---
        if reproj_error is not None:
            self._reproj_errors.append(reproj_error)

        if homography is None:
            # No usable H -- reset temporal state for H-diff and acceleration.
            self._prev_H = None
            self._prev_positions = None
            self._prev_velocity = None
            return

        # --- H-difference (Frobenius norm) ---
        if self._prev_H is not None:
            diff = np.linalg.norm(homography - self._prev_H, ord="fro")
            self._hdiff_norms.append(float(diff))
        self._prev_H = homography.copy()

        # --- Acceleration (jitter metric) ---
        projected = cv2.perspectiveTransform(self.reference_points, homography)
        current_positions = projected.reshape(-1, 2)

        if self._prev_positions is not None:
            current_velocity = current_positions - self._prev_positions
            if self._prev_velocity is not None:
                acceleration = current_velocity - self._prev_velocity
                per_point_mag = np.linalg.norm(acceleration, axis=1)
                self._accel_magnitudes.append(float(np.mean(per_point_mag)))
            self._prev_velocity = current_velocity
        else:
            self._prev_velocity = None

        self._prev_positions = current_positions

    def reset_temporal(self) -> None:
        """Reset temporal state (H-diff, acceleration) on scene cut.

        Coverage counters and reprojection errors are preserved across cuts.
        """
        self._prev_H = None
        self._prev_positions = None
        self._prev_velocity = None

    def report(self) -> StabilityReport | None:
        """Compute and return the stability report.

        Returns ``None`` if no frames have been recorded.
        """
        if self._total_frames == 0:
            return None

        def _stats(values: list[float]) -> tuple[float, float, float]:
            if not values:
                return 0.0, 0.0, 0.0
            arr = np.array(values)
            return float(np.mean(arr)), float(np.percentile(arr, 95)), float(np.max(arr))

        r_mean, r_p95, r_max = _stats(self._reproj_errors)
        h_mean, h_p95, h_max = _stats(self._hdiff_norms)
        a_mean, a_p95, a_max = _stats(self._accel_magnitudes)

        return StabilityReport(
            reproj_mean=r_mean,
            reproj_p95=r_p95,
            reproj_max=r_max,
            hdiff_mean=h_mean,
            hdiff_p95=h_p95,
            hdiff_max=h_max,
            accel_mean=a_mean,
            accel_p95=a_p95,
            accel_max=a_max,
            total_frames=self._total_frames,
            accepted=self._accepted,
            held=self._held,
            rejected=self._rejected,
            cuts=self._cuts,
        )

    def write_json(self, path: str | Path) -> None:
        """Write the stability report to a JSON file."""
        rpt = self.report()
        if rpt is None:
            logger.warning("No frames recorded -- skipping JSON write.")
            return
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rpt.to_dict(), f, indent=2)
        logger.info("Stability report written to %s", out_path)
