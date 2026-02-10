"""Abstract interface for occlusion maskers.

Every masker must implement :meth:`mask`, which takes a single video
frame and returns an :class:`OcclusionMaskerResult` dict containing a
per-pixel occlusion mask, a confidence score, and a debug payload.

Design notes
------------
- Mirrors the :class:`CourtCalibrator` / :class:`CalibrationResult`
  pattern: ABC + TypedDict for type-safe, dict-ergonomic results.
- The mask uses ``1.0 = foreground occluder`` convention so that the
  compositor can compute ``effective_alpha = ad_alpha * (1 - occ_mask)``.
- Soft masks (values between 0 and 1) are allowed for anti-aliased edges.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np


class OcclusionMaskerResult(TypedDict):
    """Typed dictionary returned by :meth:`OcclusionMasker.mask`.

    Attributes
    ----------
    mask : np.ndarray
        ``(H, W)`` float32 array in ``[0, 1]``.  ``1.0`` means
        "foreground occluder" (e.g. a player), ``0.0`` means background.
    conf : float
        Summary confidence.  Semantics are implementation-defined (e.g.
        mean detection score, or ``1.0`` when not meaningful).
    debug : dict[str, Any]
        Free-form debug payload (instance count, per-instance scores,
        timing, overlay data, etc.).
    """

    mask: np.ndarray
    conf: float
    debug: dict[str, Any]


class OcclusionMasker(ABC):
    """Abstract base class for occlusion-mask estimators.

    Subclasses must implement :meth:`mask`.
    """

    @abstractmethod
    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Generate an occlusion mask for a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image as an ``(H, W, 3)`` BGR array (OpenCV convention).

        Returns
        -------
        OcclusionMaskerResult
            A dict with keys ``mask``, ``conf``, ``debug``.
        """
        ...
