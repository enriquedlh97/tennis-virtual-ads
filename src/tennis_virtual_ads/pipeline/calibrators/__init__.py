"""Court calibrators — homography estimation from broadcast frames."""

from tennis_virtual_ads.pipeline.calibrators.base import CalibrationResult, CourtCalibrator
from tennis_virtual_ads.pipeline.calibrators.dummy import DummyCalibrator

# TennisCourtDetectorCalibrator is NOT imported here by default because it
# requires torch (heavy optional dependency).  Import it explicitly:
#   from tennis_virtual_ads.pipeline.calibrators.tennis_court_detector import (
#       TennisCourtDetectorCalibrator,
#   )

__all__ = ["CalibrationResult", "CourtCalibrator", "DummyCalibrator"]
