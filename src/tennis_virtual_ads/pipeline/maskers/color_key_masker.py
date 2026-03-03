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

Morphological cleanup (dilate/erode) is handled downstream by
:class:`MaskSmoother` — this masker intentionally stays pure-keying so that
each stage has a single responsibility.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

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

        logger.info(
            "ColorKeyMasker ready - HSV range: H[%d-%d] S[%d-%d] V[%d-%d]",
            *self._range,
        )

    # ------------------------------------------------------------------ mask
    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        court_mask = cv2.inRange(hsv, self._lower, self._upper)  # 255 = court

        # Invert: court → 0.0, non-court (occluder) → 1.0
        occ_mask = (255 - court_mask).astype(np.float32) / 255.0

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

    # --------------------------------------------------------- reset_on_cut
    def reset_on_cut(self) -> None:
        """No-op — color keying is stateless per-frame."""
