"""Abstract interface for court calibrators.

Every calibrator must implement :meth:`estimate`, which takes a single
broadcast frame and returns a :class:`CalibrationResult` dict containing
the estimated homography, a confidence score, optional keypoints, and
an arbitrary debug payload.

Design notes
------------
- We use :class:`~abc.ABC` so that missing implementations are caught at
  instantiation time, not at call time.
- :class:`CalibrationResult` is a :class:`~typing.TypedDict` so that
  consumers get key-level type checking without losing dict ergonomics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

import numpy as np


class CalibrationResult(TypedDict):
    """Typed dictionary returned by :meth:`CourtCalibrator.estimate`.

    Attributes
    ----------
    H : np.ndarray | None
        3x3 homography matrix (court-plane to image), or ``None`` when
        calibration fails or is unavailable.
    conf : float
        Confidence score in ``[0.0, 1.0]``.  ``0.0`` means "no
        estimate"; ``1.0`` means "fully confident".
    keypoints : np.ndarray | None
        Detected court keypoints used to compute *H*.  Shape is
        implementation-defined.  ``None`` when not applicable.
    debug : dict[str, Any]
        Free-form debug payload (overlay images, reprojection errors,
        timing info, etc.).
    """

    H: np.ndarray | None
    conf: float
    keypoints: np.ndarray | None
    debug: dict[str, Any]


class CourtCalibrator(ABC):
    """Abstract base class for court-homography estimators.

    Subclasses must implement :meth:`estimate`.
    """

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> CalibrationResult:
        """Estimate the court homography for a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image as an ``(H, W, 3)`` BGR array (OpenCV convention).

        Returns
        -------
        CalibrationResult
            A dict with keys ``H``, ``conf``, ``keypoints``, ``debug``.
        """
        ...
