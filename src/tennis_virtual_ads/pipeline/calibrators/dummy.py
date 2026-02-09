"""Dummy calibrator — always returns *no estimate*.

Useful for:
- validating the pipeline wiring end-to-end without any ML,
- as a fallback when no real calibrator is configured.
"""

from __future__ import annotations

import numpy as np

from tennis_virtual_ads.pipeline.calibrators.base import CalibrationResult, CourtCalibrator


class DummyCalibrator(CourtCalibrator):
    """Calibrator that unconditionally returns ``H=None, conf=0.0``.

    No computation is performed; every call is O(1).
    """

    def estimate(self, frame: np.ndarray) -> CalibrationResult:
        """Return an empty calibration result regardless of *frame*."""
        return CalibrationResult(
            H=None,
            conf=0.0,
            keypoints=None,
            debug={},
        )
