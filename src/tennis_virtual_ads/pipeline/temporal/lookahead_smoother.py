"""Non-causal lookahead smoother for near-zero-jitter homography output.

The key insight is that a small lookahead (~10 frames = 333ms at 30fps) is
acceptable for live broadcast and enables **non-causal** smoothing that is
dramatically better than any causal filter (EMA, Kalman).

How it works
------------
1. Accumulate homography observations into a growing list.
2. Once there are enough *future* frames for the next-to-output frame,
   smooth it using up to ``lookahead`` past and ``lookahead`` future frames.
3. The output delay is ``lookahead`` frames (~333ms at 30fps for default 10).
4. At end of video, :meth:`flush` drains remaining frames with shrinking
   future context.

The filter is Savitzky-Golay applied independently to each decomposed
camera parameter ``[pan, tilt, focal]``.  Savitzky-Golay fits a local
polynomial (order 3) to the data in the window -- it smooths noise while
preserving the shape of genuine camera motion (pans, tilts, zooms).

Scene cuts split the smoothing window so that no smoothing crosses a cut
boundary.
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
class _FrameEntry:
    """One buffered frame's homography data."""

    homography: np.ndarray | None
    params: np.ndarray | None  # [pan, tilt, focal] or None if decomposition failed
    is_cut: bool


def _savgol_smooth_params(
    params_sequence: list[np.ndarray],
    polyorder: int = 3,
    center_idx: int | None = None,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a sequence of camera params.

    Returns the smoothed params for the center frame (or *center_idx* if
    specified).

    Parameters
    ----------
    params_sequence : list[np.ndarray]
        Each element is ``[pan, tilt, focal]``.
    polyorder : int
        Polynomial order for Savitzky-Golay.
    center_idx : int | None
        Index of the frame to return.  Defaults to the middle element.

    Returns
    -------
    np.ndarray
        Smoothed ``[pan, tilt, focal]`` for the target frame.
    """
    from scipy.signal import savgol_filter

    n = len(params_sequence)
    if center_idx is None:
        center_idx = n // 2

    # Stack into (N, 3) array.
    arr = np.array(params_sequence)  # (N, 3)

    # Window length must be odd and <= n.
    window_length = n if n % 2 == 1 else n - 1
    window_length = max(window_length, polyorder + 2)
    if window_length > n:
        # Not enough data for the requested polyorder -- return raw.
        return np.array(arr[center_idx], copy=True)

    # Clamp polyorder if window is small.
    effective_polyorder = min(polyorder, window_length - 1)

    # Apply savgol independently per channel.
    smoothed = savgol_filter(arr, window_length, effective_polyorder, axis=0)

    return np.array(smoothed[center_idx], copy=True)


class LookaheadSmoother:
    """Non-causal homography smoother with fixed-latency lookahead buffer.

    Each :meth:`push` call appends one frame.  Once there are ``lookahead``
    future frames available for the next-to-output frame, a smoothed
    homography is returned.  This gives a consistent 1:1 input→output
    ratio after the initial delay, making external frame-buffer
    synchronisation trivial.

    Parameters
    ----------
    lookahead : int
        Number of future frames to buffer.  Output delay equals this
        value.  Default ``10`` (~333ms at 30fps).
    polyorder : int
        Savitzky-Golay polynomial order.  Default ``3``.
    image_center : tuple[float, float]
        Image center for H decomposition.  Default ``(640.0, 360.0)``.
    """

    def __init__(
        self,
        lookahead: int = 10,
        polyorder: int = 3,
        image_center: tuple[float, float] = (640.0, 360.0),
    ) -> None:
        if lookahead < 1:
            raise ValueError(f"lookahead must be >= 1, got {lookahead}")
        self._lookahead = lookahead
        self._polyorder = polyorder
        self._image_center = image_center

        self._entries: list[_FrameEntry] = []
        self._next_output: int = 0  # Index of next frame to output.

    @property
    def lookahead(self) -> int:
        """The lookahead value (frames of delay)."""
        return self._lookahead

    @property
    def delay(self) -> int:
        """Alias for :attr:`lookahead` (output delay in frames)."""
        return self._lookahead

    def push(
        self,
        homography: np.ndarray | None,
        is_cut: bool = False,
    ) -> np.ndarray | None:
        """Add one frame's homography.  Returns a smoothed H when the
        next-to-output frame has enough future context.

        After the initial ``lookahead``-frame delay, every call returns
        exactly one smoothed H (1:1 correspondence with an external
        frame buffer).

        Parameters
        ----------
        homography : np.ndarray | None
            Raw (or tracked) homography for this frame.  ``None`` when
            calibration failed and no hold-H is available.
        is_cut : bool
            Whether a scene cut was detected at this frame.

        Returns
        -------
        np.ndarray | None
            Smoothed homography for the frame at index
            ``self._next_output``, or ``None`` if not enough future
            context has accumulated yet.
        """
        params: np.ndarray | None = None
        if homography is not None:
            params = _decompose_homography_to_params(homography, self._image_center)

        self._entries.append(_FrameEntry(homography=homography, params=params, is_cut=is_cut))

        # Check if the next-to-output frame has enough future context.
        # It needs at least `lookahead` frames after it.
        future_available = len(self._entries) - 1 - self._next_output
        if future_available >= self._lookahead:
            result = self._smooth_entry(self._next_output)
            self._next_output += 1
            return result

        return None

    def flush(self) -> list[np.ndarray | None]:
        """Drain remaining buffered frames at end of video.

        Returns a list of smoothed homographies for each remaining
        un-output frame (from oldest to newest).  These frames have
        less than ``lookahead`` future context, so the smoothing window
        shrinks progressively.
        """
        results: list[np.ndarray | None] = []
        while self._next_output < len(self._entries):
            result = self._smooth_entry(self._next_output)
            results.append(result)
            self._next_output += 1
        return results

    def reset(self) -> None:
        """Clear all state."""
        self._entries.clear()
        self._next_output = 0

    # ------------------------------------------------------------------
    # Internal smoothing logic
    # ------------------------------------------------------------------

    def _smooth_entry(self, target_idx: int) -> np.ndarray | None:
        """Smooth the frame at *target_idx* using available context.

        The smoothing window extends up to ``lookahead`` frames in each
        direction, bounded by scene cuts and the available data range.
        """
        n = len(self._entries)
        if target_idx < 0 or target_idx >= n:
            return None

        target_entry = self._entries[target_idx]

        # If the target has no valid H, return None.
        if target_entry.homography is None or target_entry.params is None:
            return None

        # Determine context window: up to lookahead in each direction.
        window_start = max(0, target_idx - self._lookahead)
        window_end = min(n - 1, target_idx + self._lookahead)

        # Narrow window to not cross scene cuts.
        # Expand left: stop if a cut is encountered (a cut at index i
        # means i is the first frame of the new segment).
        for i in range(target_idx - 1, window_start - 1, -1):
            if self._entries[i + 1].is_cut:
                window_start = i + 1
                break

        # Expand right: stop if a cut is encountered.
        for i in range(target_idx + 1, window_end + 1):
            if self._entries[i].is_cut:
                window_end = i - 1
                break

        # Collect valid params in the window.
        valid_indices: list[int] = []
        valid_params: list[np.ndarray] = []
        for i in range(window_start, window_end + 1):
            entry = self._entries[i]
            if entry.params is not None:
                valid_indices.append(i)
                valid_params.append(entry.params)

        if len(valid_params) < 2:
            # Not enough context to smooth -- return the raw H.
            return (
                np.array(target_entry.homography, copy=True)
                if target_entry.homography is not None
                else None
            )

        # Find where the target sits in the valid_params list.
        try:
            target_pos_in_valid = valid_indices.index(target_idx)
        except ValueError:
            return (
                np.array(target_entry.homography, copy=True)
                if target_entry.homography is not None
                else None
            )

        smoothed_params = _savgol_smooth_params(
            valid_params,
            polyorder=self._polyorder,
            center_idx=target_pos_in_valid,
        )

        # Reconstruct H from smoothed params.
        H_smoothed = _reconstruct_homography_from_params(smoothed_params, self._image_center)
        if abs(H_smoothed[2, 2]) > 1e-12:
            H_smoothed = H_smoothed / H_smoothed[2, 2]

        return np.asarray(H_smoothed)
