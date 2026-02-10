"""Shadow-preserving "painted" compositing for court ads.

The ad inherits the court's illumination / shadow pattern so it looks
like it's painted on the surface rather than pasted on top.

Algorithm (v1)
--------------
1. **Alpha feather** -- optionally blur ``effective_alpha`` with a small
   Gaussian kernel to soften hard edges.
2. **Illumination estimation** -- convert the original frame to
   grayscale and apply a large Gaussian blur to extract the low-
   frequency illumination field.
3. **Shade normalization** -- divide the illumination field by the
   median brightness inside the ``court_mask`` region.  This produces a
   shade map where ``1.0`` = average court brightness, ``<1.0`` = shadow,
   ``>1.0`` = bright spot.  Clipped to a safe range to avoid extreme
   colours.  An optional ``shade_strength`` exponent controls intensity.
4. **Apply shading to ad** -- multiply the warped ad's RGB channels by
   the shade map so the ad darkens under shadows and brightens in hot
   spots.
5. **Composite** -- standard alpha blend with the shaded ad colours.

The function is pure OpenCV/NumPy with no external dependencies.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def painted_composite(
    frame: np.ndarray,
    warped_rgba: np.ndarray,
    effective_alpha: np.ndarray,
    court_mask: np.ndarray | None,
    shade_blur_ksize: int = 41,
    shade_strength: float = 1.0,
    alpha_feather_px: int = 3,
    shade_clip_min: float = 0.6,
    shade_clip_max: float = 1.4,
) -> dict[str, Any]:
    """Composite a warped ad onto *frame* with shadow-preserving blending.

    Mutates *frame* in-place and returns a debug dict.

    Parameters
    ----------
    frame : np.ndarray
        ``(H, W, 3)`` uint8 BGR video frame (mutated in-place).
    warped_rgba : np.ndarray
        ``(H, W, 4)`` uint8 BGRA warped ad image.
    effective_alpha : np.ndarray
        ``(H, W)`` float32 in ``[0, 1]``.  Already includes occlusion
        (``warped_mask * (1 - occ_mask)``).
    court_mask : np.ndarray | None
        ``(H, W)`` uint8 mask where ``255`` = court surface pixels.
        Used to compute median court brightness for shade normalisation.
        When ``None``, falls back to the ad's own alpha footprint.
    shade_blur_ksize : int
        Gaussian blur kernel size for illumination extraction.  Must be
        odd.  Larger = smoother illumination field.  Default ``41``.
    shade_strength : float
        Exponent applied to the shade map.  ``1.0`` = natural;
        ``>1.0`` = exaggerated shadow effect.  Default ``1.0``.
    alpha_feather_px : int
        Gaussian blur radius applied to ``effective_alpha`` for edge
        softening.  ``0`` disables feathering.  Default ``3``.
    shade_clip_min : float
        Minimum shade value (prevents extreme darkening).  Default ``0.6``.
    shade_clip_max : float
        Maximum shade value (prevents extreme brightening).  Default ``1.4``.

    Returns
    -------
    dict[str, Any]
        Debug payload with optional ``shade_map`` (the normalised shade
        field as float32 in ``[shade_clip_min, shade_clip_max]``).
    """
    debug: dict[str, Any] = {}

    # --- 1. Alpha feather (optional) --------------------------------------
    alpha = effective_alpha
    if alpha_feather_px > 0:
        feather_ksize = 2 * alpha_feather_px + 1  # ensure odd
        alpha = cv2.GaussianBlur(alpha, (feather_ksize, feather_ksize), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    # --- 2. Illumination estimation from original frame -------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Ensure kernel size is odd.
    blur_ksize = shade_blur_ksize if shade_blur_ksize % 2 == 1 else shade_blur_ksize + 1
    illumination = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # --- 3. Shade normalisation relative to court region ------------------
    # Compute reference brightness from the court surface.  If no court
    # mask is available, use the ad footprint as a rough proxy.
    if court_mask is not None and np.any(court_mask > 0):
        court_pixels = illumination[court_mask > 0]
        median_illumination = float(np.median(court_pixels))
    elif np.any(alpha > 0.01):
        # Fallback: use the ad's own alpha footprint.
        ad_region_pixels = illumination[alpha > 0.01]
        median_illumination = float(np.median(ad_region_pixels))
    else:
        median_illumination = 128.0  # safe default

    # Avoid division by zero.
    median_illumination = max(median_illumination, 1.0)

    shade_map = illumination / median_illumination

    # Apply shade strength exponent.
    if shade_strength != 1.0:
        shade_map = np.power(shade_map, shade_strength)

    # Clip to safe range.
    shade_map = np.clip(shade_map, shade_clip_min, shade_clip_max)

    debug["shade_map"] = shade_map

    # --- 4. Apply shading to warped ad colours ----------------------------
    warped_bgr = warped_rgba[:, :, :3].astype(np.float32) / 255.0
    shaded_ad = warped_bgr * shade_map[:, :, np.newaxis]
    shaded_ad = np.clip(shaded_ad, 0.0, 1.0)

    # --- 5. Composite: out = frame*(1-alpha) + shaded_ad*alpha ------------
    alpha_3channel = alpha[:, :, np.newaxis]
    frame_float = frame.astype(np.float32)

    blended = frame_float * (1.0 - alpha_3channel) + (shaded_ad * 255.0) * alpha_3channel
    np.copyto(frame, blended.astype(np.uint8))

    return debug
