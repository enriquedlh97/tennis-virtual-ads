"""Temporal mask smoother to reduce frame-to-frame occlusion mask flicker.

Pipeline per frame::

    raw_mask → morphological close → max(current, prev_raw)

Keeps the mask **binary (float32 {0, 1})** throughout.
No EMA blend (which erodes mask edges) — just morphological cleanup
and single-frame dropout protection.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


class MaskSmoother:
    """Morphological smoother for occlusion masks.

    Parameters
    ----------
    close_px : int
        Morphological close kernel radius (fills arm-body gaps).
        ``0`` disables closing.  Default ``7``.
    """

    def __init__(
        self,
        close_px: int = 7,
    ) -> None:
        self._close_px = close_px

        self._prev_raw: np.ndarray | None = None

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

        return mask

    def reset(self) -> None:
        """Clear temporal state (call on scene cuts)."""
        self._prev_raw = None
