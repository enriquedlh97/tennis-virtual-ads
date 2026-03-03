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

The filter is Savitzky-Golay applied independently to each of the 8
free entries of the normalised homography matrix (H[2,2]=1).  This
avoids lossy decomposition into camera parameters.

Scene cuts split the smoothing window so that no smoothing crosses a cut
boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _normalise_h(H: np.ndarray) -> np.ndarray:
    """Normalise a 3x3 homography so that H[2,2] = 1."""
    h = np.asarray(H, dtype=np.float64).ravel()
    if abs(h[8]) > 1e-15:
        h = h / h[8]
    return h  # flat 9-vector


@dataclass
class _FrameEntry:
    """One buffered frame's homography data."""

    homography: np.ndarray | None
    h_vec: np.ndarray | None  # flat 9-vector (normalised) or None
    is_cut: bool


def _savgol_smooth(
    vectors: list[np.ndarray],
    polyorder: int = 3,
    center_idx: int | None = None,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a sequence of vectors.

    Returns the smoothed vector for the center frame (or *center_idx*).
    """
    from scipy.signal import savgol_filter

    n = len(vectors)
    if center_idx is None:
        center_idx = n // 2

    arr = np.array(vectors)  # (N, D)

    # Window length must be odd and <= n.
    window_length = n if n % 2 == 1 else n - 1
    window_length = max(window_length, polyorder + 2)
    if window_length > n:
        return np.array(arr[center_idx], copy=True)

    effective_polyorder = min(polyorder, window_length - 1)
    smoothed = savgol_filter(arr, window_length, effective_polyorder, axis=0)
    return np.array(smoothed[center_idx], copy=True)


class LookaheadSmoother:
    """Non-causal homography smoother with fixed-latency lookahead buffer.

    Each :meth:`push` call appends one frame.  Once there are ``lookahead``
    future frames available for the next-to-output frame, a smoothed
    homography is returned.  This gives a consistent 1:1 input/output
    ratio after the initial delay, making external frame-buffer
    synchronisation trivial.

    Parameters
    ----------
    lookahead : int
        Number of future frames to buffer.  Output delay equals this
        value.  Default ``10`` (~333ms at 30fps).
    polyorder : int
        Savitzky-Golay polynomial order.  Default ``3``.
    """

    def __init__(
        self,
        lookahead: int = 10,
        polyorder: int = 3,
    ) -> None:
        if lookahead < 1:
            raise ValueError(f"lookahead must be >= 1, got {lookahead}")
        self._lookahead = lookahead
        self._polyorder = polyorder

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
        h_vec: np.ndarray | None = None
        if homography is not None:
            h_vec = _normalise_h(homography)

        self._entries.append(_FrameEntry(homography=homography, h_vec=h_vec, is_cut=is_cut))

        # Check if the next-to-output frame has enough future context.
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
        if target_entry.homography is None or target_entry.h_vec is None:
            return None

        # Determine context window: up to lookahead in each direction.
        window_start = max(0, target_idx - self._lookahead)
        window_end = min(n - 1, target_idx + self._lookahead)

        # Narrow window to not cross scene cuts.
        for i in range(target_idx - 1, window_start - 1, -1):
            if self._entries[i + 1].is_cut:
                window_start = i + 1
                break

        for i in range(target_idx + 1, window_end + 1):
            if self._entries[i].is_cut:
                window_end = i - 1
                break

        # Collect valid H vectors in the window.
        valid_indices: list[int] = []
        valid_vecs: list[np.ndarray] = []
        for i in range(window_start, window_end + 1):
            entry = self._entries[i]
            if entry.h_vec is not None:
                valid_indices.append(i)
                valid_vecs.append(entry.h_vec)

        if len(valid_vecs) < 2:
            # Not enough context — return raw H.
            return (
                np.array(target_entry.homography, copy=True)
                if target_entry.homography is not None
                else None
            )

        # Find where the target sits in the valid list.
        try:
            target_pos = valid_indices.index(target_idx)
        except ValueError:
            return (
                np.array(target_entry.homography, copy=True)
                if target_entry.homography is not None
                else None
            )

        smoothed_vec = _savgol_smooth(
            valid_vecs,
            polyorder=self._polyorder,
            center_idx=target_pos,
        )

        # Reshape back to 3x3 and normalise.
        H_smoothed = smoothed_vec.reshape(3, 3)
        if abs(H_smoothed[2, 2]) > 1e-12:
            H_smoothed = H_smoothed / H_smoothed[2, 2]

        return np.asarray(H_smoothed, dtype=np.float64)
