"""Temporal mask smoother to reduce frame-to-frame occlusion mask flicker.

Pipeline per frame::

    raw_mask → morphological close → max(current, prev_raw) → EMA blend → hysteresis threshold

The mask stays **soft (float32 [0,1])** throughout — no hard binarisation.
The compositor already handles continuous alpha values.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


class MaskSmoother:
    """EMA-based temporal smoother for occlusion masks.

    Parameters
    ----------
    alpha : float
        EMA blending factor in ``(0, 1]``.  Higher = more responsive,
        lower = smoother.  Default ``0.5``.
    close_px : int
        Morphological close kernel radius (fills arm-body gaps).
        ``0`` disables closing.  Default ``7``.
    hysteresis_low : float
        Pixels below this value after EMA are zeroed.  Default ``0.3``.
    hysteresis_high : float
        Pixels above this value after EMA are set to 1.0.  Default ``0.6``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        close_px: int = 7,
        hysteresis_low: float = 0.3,
        hysteresis_high: float = 0.6,
    ) -> None:
        self._alpha = alpha
        self._close_px = close_px
        self._hysteresis_low = hysteresis_low
        self._hysteresis_high = hysteresis_high

        self._prev_raw: np.ndarray | None = None
        self._prev_smooth: np.ndarray | None = None

    def update(self, raw_mask: np.ndarray) -> NDArray[Any]:
        """Smooth *raw_mask* and return the stabilised result.

        Parameters
        ----------
        raw_mask : np.ndarray
            ``(H, W)`` float32 mask in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            ``(H, W)`` float32 smoothed mask in ``[0, 1]``.
        """
        mask: NDArray[np.floating[Any]] = raw_mask.astype(np.float32, copy=True)

        # 1. Morphological close — fill holes within the mask.
        if self._close_px > 0 and np.any(mask > 0):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self._close_px + 1, 2 * self._close_px + 1),
            )
            mask = np.asarray(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), dtype=np.float32)

        # 2. Single-frame dropout protection — union with previous raw mask.
        if self._prev_raw is not None and self._prev_raw.shape == mask.shape:
            mask = np.maximum(mask, self._prev_raw).astype(np.float32)
        self._prev_raw = raw_mask.astype(np.float32, copy=True)

        # 3. EMA blend.
        if self._prev_smooth is None or self._prev_smooth.shape != mask.shape:
            smooth = mask
        else:
            smooth = (self._alpha * mask + (1.0 - self._alpha) * self._prev_smooth).astype(
                np.float32
            )
        self._prev_smooth = smooth.copy()

        # 4. Hysteresis threshold — prevent ghost trails, keep soft edges.
        out: NDArray[np.floating[Any]] = smooth.copy()
        out[smooth >= self._hysteresis_high] = 1.0
        out[smooth < self._hysteresis_low] = 0.0

        return out

    def reset(self) -> None:
        """Clear temporal state (call on scene cuts)."""
        self._prev_raw = None
        self._prev_smooth = None
