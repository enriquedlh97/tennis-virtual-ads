"""HSV court-color keying masker.

Professional broadcast systems (Supponor, Viz Arena) use color keying rather
than ML segmentation because:

1. **Motion-blur robust** — color is preserved even in heavily blurred pixels,
   whereas instance-segmentation models lose boundary precision.
2. **Sub-1 ms per frame** — pure OpenCV, no GPU required.
3. **Deterministic** — no model-dependent false negatives.

The idea is simple: tennis courts are a known, nearly uniform color (blue
hard-court, green grass/hard, clay orange).  Any pixel whose HSV value falls
*inside* the court range is background (mask = 0); everything else is a
potential occluder (mask = 1).

When ``softness > 0``, the binary threshold is replaced with a continuous HSV
distance function that produces soft [0, 1] alpha at court-color boundaries.
This avoids ad bleed-through on motion-blurred limbs, where pixels are a blend
of player and court color.  An optional guided filter (requires
``opencv-contrib-python``) refines the soft mask to be edge-aware.

Morphological cleanup (dilate/erode) is handled downstream by
:class:`MaskSmoother` — this masker intentionally stays pure-keying so that
each stage has a single responsibility.
"""

from __future__ import annotations

import logging
import warnings
from typing import NamedTuple

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

try:
    from cv2 import ximgproc as _ximgproc

    _HAS_GUIDED_FILTER = True
except ImportError:
    _ximgproc = None  # type: ignore[assignment]
    _HAS_GUIDED_FILTER = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HSV range helper
# ---------------------------------------------------------------------------


class HSVRange(NamedTuple):
    """Inclusive HSV range (OpenCV convention: H 0-179, S/V 0-255)."""

    h_low: int
    h_high: int
    s_low: int
    s_high: int
    v_low: int
    v_high: int


# ---------------------------------------------------------------------------
# Presets for common court surfaces
# ---------------------------------------------------------------------------

COURT_COLOR_PRESETS: dict[str, HSVRange] = {
    "blue_hard": HSVRange(h_low=95, h_high=125, s_low=40, s_high=255, v_low=30, v_high=255),
    "green_hard": HSVRange(h_low=35, h_high=85, s_low=40, s_high=255, v_low=30, v_high=255),
    "clay": HSVRange(h_low=5, h_high=25, s_low=50, s_high=255, v_low=40, v_high=255),
}


# ---------------------------------------------------------------------------
# Masker
# ---------------------------------------------------------------------------


class ColorKeyMasker(OcclusionMasker):
    """Court-color keying masker using HSV thresholding.

    Pixels whose HSV values fall within the court-color range are classified
    as background (0.0); everything else is foreground/occluder (1.0).

    Parameters
    ----------
    preset : str or None
        One of ``COURT_COLOR_PRESETS`` keys.  Individual ``h_low`` … ``v_high``
        overrides are applied on top.
    h_low, h_high, s_low, s_high, v_low, v_high : int or None
        Manual HSV overrides.  Any non-None value replaces the corresponding
        preset value.
    softness : float
        Falloff width in HSV units.  ``0`` = binary (current behavior).
        Typical: 10-20.  Higher values produce wider soft transitions.
    guided_filter_radius : int
        Spatial window for edge-aware guided filter refinement.
    guided_filter_eps : float
        Regularization for guided filter (smaller = sharper edges).
    """

    def __init__(
        self,
        preset: str | None = "blue_hard",
        *,
        h_low: int | None = None,
        h_high: int | None = None,
        s_low: int | None = None,
        s_high: int | None = None,
        v_low: int | None = None,
        v_high: int | None = None,
        softness: float = 0.0,
        guided_filter_radius: int = 8,
        guided_filter_eps: float = 0.01,
    ) -> None:
        # Start from preset (or zeros) ----------------------------------
        if preset is not None:
            if preset not in COURT_COLOR_PRESETS:
                valid = ", ".join(sorted(COURT_COLOR_PRESETS))
                raise ValueError(f"Unknown court color preset '{preset}'. Available: {valid}")
            base = COURT_COLOR_PRESETS[preset]
        else:
            base = HSVRange(0, 179, 0, 255, 0, 255)

        # Apply manual overrides ----------------------------------------
        self._range = HSVRange(
            h_low=h_low if h_low is not None else base.h_low,
            h_high=h_high if h_high is not None else base.h_high,
            s_low=s_low if s_low is not None else base.s_low,
            s_high=s_high if s_high is not None else base.s_high,
            v_low=v_low if v_low is not None else base.v_low,
            v_high=v_high if v_high is not None else base.v_high,
        )

        # Pre-compute numpy bounds for cv2.inRange ----------------------
        self._lower = np.array(
            [self._range.h_low, self._range.s_low, self._range.v_low], dtype=np.uint8
        )
        self._upper = np.array(
            [self._range.h_high, self._range.s_high, self._range.v_high], dtype=np.uint8
        )

        # Soft-keying parameters ----------------------------------------
        self._softness = float(softness)
        self._guided_filter_radius = guided_filter_radius
        self._guided_filter_eps = guided_filter_eps

        # Pre-compute float32 bounds for the soft path
        self._h_low_f = float(self._range.h_low)
        self._h_high_f = float(self._range.h_high)
        self._s_low_f = float(self._range.s_low)
        self._s_high_f = float(self._range.s_high)
        self._v_low_f = float(self._range.v_low)
        self._v_high_f = float(self._range.v_high)

        self._use_guided_filter = self._softness > 0 and _HAS_GUIDED_FILTER
        if self._softness > 0 and not _HAS_GUIDED_FILTER:
            warnings.warn(
                "Guided filter requested but cv2.ximgproc is not available "
                "(install opencv-contrib-python). Falling back to soft-only keying.",
                stacklevel=2,
            )

        mode = "binary"
        if self._softness > 0:
            mode = "soft+guided" if self._use_guided_filter else "soft"
        logger.info(
            "ColorKeyMasker ready - HSV range: H[%d-%d] S[%d-%d] V[%d-%d], mode=%s, softness=%.1f",
            *self._range,
            mode,
            self._softness,
        )

    # ------------------------------------------------------------------ mask
    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        occ_mask = self._mask_binary(hsv) if self._softness <= 0 else self._mask_soft(hsv, frame)

        # Sanity check: warn if most of the frame is classified as occluder
        occluder_ratio = float(occ_mask.mean())
        if occluder_ratio > 0.80:
            logger.warning(
                "ColorKeyMasker: %.0f%% of pixels classified as occluder — "
                "HSV range may be wrong for this court surface.",
                occluder_ratio * 100,
            )

        return OcclusionMaskerResult(
            mask=occ_mask,
            conf=1.0,
            debug={"occluder_ratio": occluder_ratio},
        )

    def _mask_binary(self, hsv: np.ndarray) -> np.ndarray:
        """Original binary path — cv2.inRange."""
        court_mask = cv2.inRange(hsv, self._lower, self._upper)  # 255 = court
        return (255 - court_mask).astype(np.float32) / 255.0

    def _mask_soft(self, hsv: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Soft HSV distance keying with optional guided filter."""
        hsv_f = hsv.astype(np.float32)
        h, s, v = hsv_f[:, :, 0], hsv_f[:, :, 1], hsv_f[:, :, 2]

        # S/V distance: zero inside range, positive outside ----------------
        d_s = np.clip(self._s_low_f - s, 0.0, None) + np.clip(s - self._s_high_f, 0.0, None)
        d_v = np.clip(self._v_low_f - v, 0.0, None) + np.clip(v - self._v_high_f, 0.0, None)

        # H distance with wrap-around (H range is 0-179) ------------------
        h_low = self._h_low_f
        h_high = self._h_high_f

        if h_low <= h_high:
            # Normal range: inside = [h_low, h_high]
            d_h = np.clip(h_low - h, 0.0, None) + np.clip(h - h_high, 0.0, None)
            # Wrap-aware: also check distance going the other way around
            d_h_wrap = 180.0 - d_h
            d_h = np.minimum(d_h, d_h_wrap)
            # Zero out for pixels inside the range
            inside_h = (h >= h_low) & (h <= h_high)
            d_h[inside_h] = 0.0
        else:
            # Wrapped range: inside = [h_low, 179] U [0, h_high]
            inside_h = (h >= h_low) | (h <= h_high)
            # Distance to nearest boundary
            dist_to_low = h_low - h  # positive when h < h_low
            dist_to_high = h - h_high  # positive when h > h_high
            # Wrap distances
            dist_to_low_wrap = h + (180.0 - h_low)
            dist_to_high_wrap = h_high + (180.0 - h)
            d_to_low = np.minimum(np.abs(dist_to_low), dist_to_low_wrap)
            d_to_high = np.minimum(np.abs(dist_to_high), dist_to_high_wrap)
            d_h = np.minimum(d_to_low, d_to_high)
            d_h[inside_h] = 0.0

        # Combine into occluder alpha --------------------------------------
        inv_soft = 1.0 / self._softness
        occ = np.sqrt((d_h * inv_soft) ** 2 + (d_s * inv_soft) ** 2 + (d_v * inv_soft) ** 2)
        occ = np.clip(occ, 0.0, 1.0).astype(np.float32)

        # Optional guided filter refinement --------------------------------
        if self._use_guided_filter:
            guide = frame.astype(np.float32) / 255.0
            occ = _ximgproc.guidedFilter(
                guide=guide,
                src=occ,
                radius=self._guided_filter_radius,
                eps=self._guided_filter_eps,
            )
            occ = np.clip(occ, 0.0, 1.0)

        return np.asarray(occ)

    # --------------------------------------------------------- reset_on_cut
    def reset_on_cut(self) -> None:
        """No-op — color keying is stateless per-frame."""
