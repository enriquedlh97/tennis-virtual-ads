"""Two-pass full-video smoother for maximum-quality post-processing.

Same Savitzky-Golay approach as :class:`LookaheadSmoother` but applied to
the **entire** video at once:

- **Pass 1:** Collect all homography observations (no rendering).
- **Smooth:** Apply scene-cut-aware Savitzky-Golay to the full sequence.
- **Pass 2:** Caller re-reads the video and uses the smoothed homographies.

This gives the best possible smoothing (full future + past context) but
requires two passes over the video (2x processing time).  Use for offline
post-processing when maximum quality is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from tennis_virtual_ads.pipeline.temporal.homography_stabilizer import (
    _decompose_homography_to_params,
    _reconstruct_homography_from_params,
)

logger = logging.getLogger(__name__)


@dataclass
class _FrameRecord:
    """One frame's data collected during pass 1."""

    homography: np.ndarray | None
    params: np.ndarray | None
    is_cut: bool


class TwoPassSmoother:
    """Full-video two-pass homography smoother.

    Parameters
    ----------
    window_length : int
        Savitzky-Golay window length (must be odd).  Default ``31``.
    polyorder : int
        Polynomial order.  Default ``3``.
    image_center : tuple[float, float]
        Image center for H decomposition.  Default ``(640.0, 360.0)``.
    """

    def __init__(
        self,
        window_length: int = 31,
        polyorder: int = 3,
        image_center: tuple[float, float] = (640.0, 360.0),
    ) -> None:
        if window_length < 3 or window_length % 2 == 0:
            raise ValueError(f"window_length must be odd and >= 3, got {window_length}")
        self._window_length = window_length
        self._polyorder = polyorder
        self._image_center = image_center

        self._records: list[_FrameRecord] = []

    # ------------------------------------------------------------------
    # Pass 1: collect
    # ------------------------------------------------------------------

    def collect(
        self,
        homography: np.ndarray | None,
        is_cut: bool = False,
    ) -> None:
        """Record one frame's homography during pass 1.

        Parameters
        ----------
        homography : np.ndarray | None
            Raw (or tracked) homography, or ``None`` on calibration failure.
        is_cut : bool
            Whether a scene cut was detected at this frame.
        """
        params: np.ndarray | None = None
        if homography is not None:
            params = _decompose_homography_to_params(homography, self._image_center)

        self._records.append(_FrameRecord(homography=homography, params=params, is_cut=is_cut))

    @property
    def frame_count(self) -> int:
        """Number of frames collected so far."""
        return len(self._records)

    # ------------------------------------------------------------------
    # Smooth: process entire sequence
    # ------------------------------------------------------------------

    def smooth(self) -> list[np.ndarray | None]:
        """Smooth all collected homographies and return the full sequence.

        Returns
        -------
        list[np.ndarray | None]
            One smoothed homography per frame, or ``None`` where no valid
            H existed.
        """
        from scipy.signal import savgol_filter

        n = len(self._records)
        if n == 0:
            return []

        # Identify contiguous segments separated by cuts.
        segments: list[tuple[int, int]] = []  # (start, end_exclusive)
        seg_start = 0
        for i in range(n):
            if self._records[i].is_cut and i > seg_start:
                segments.append((seg_start, i))
                seg_start = i
        segments.append((seg_start, n))

        # Prepare output.
        result: list[np.ndarray | None] = [None] * n

        for seg_start, seg_end in segments:
            # Collect valid indices and params within this segment.
            valid_indices: list[int] = []
            valid_params: list[np.ndarray] = []

            for i in range(seg_start, seg_end):
                rec = self._records[i]
                if rec.params is not None:
                    valid_indices.append(i)
                    valid_params.append(rec.params)

            if len(valid_params) == 0:
                # No valid frames in this segment.
                continue

            if len(valid_params) == 1:
                # Single valid frame -- no smoothing possible.
                idx = valid_indices[0]
                result[idx] = self._records[idx].homography
                continue

            # Stack into (M, 3) array.
            arr = np.array(valid_params)

            # Determine effective window length.
            wl = min(self._window_length, len(valid_params))
            if wl % 2 == 0:
                wl -= 1
            wl = max(wl, self._polyorder + 2)
            if wl > len(valid_params):
                # Can't smooth with this polyorder -- output raw.
                for _vi, idx in enumerate(valid_indices):
                    result[idx] = self._records[idx].homography
                continue

            effective_po = min(self._polyorder, wl - 1)

            # Apply savgol per channel.
            smoothed_arr = savgol_filter(arr, wl, effective_po, axis=0)

            # Reconstruct H for each valid frame.
            for vi, idx in enumerate(valid_indices):
                smoothed_params = smoothed_arr[vi]
                H_s = _reconstruct_homography_from_params(smoothed_params, self._image_center)
                if abs(H_s[2, 2]) > 1e-12:
                    H_s = H_s / H_s[2, 2]
                result[idx] = np.asarray(H_s)

            # For invalid frames within the segment that are surrounded
            # by valid ones, we leave them as None (the caller should
            # handle this via hold logic or skip).

        logger.info(
            "Two-pass smooth: %d frames, %d segments, window=%d, polyorder=%d",
            n,
            len(segments),
            self._window_length,
            self._polyorder,
        )

        return result

    def reset(self) -> None:
        """Clear all collected data."""
        self._records.clear()
