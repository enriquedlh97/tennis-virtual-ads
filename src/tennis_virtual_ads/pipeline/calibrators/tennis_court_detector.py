"""Court calibrator wrapping the TennisCourtDetector neural network.

This calibrator uses a pre-trained heatmap-based keypoint detection model
(BallTrackerNet architecture) to detect 14 court keypoints from a broadcast
frame.  A homography is then computed using best-of-12 configuration
selection (try each configuration of 4 reference-point correspondences,
keep the one with lowest mean reprojection error on the remaining points).

The neural network model (BallTrackerNet) is imported from the sibling
``TennisCourtDetector`` repository via ``importlib`` so we do not need to
copy the model code or pollute ``sys.path``.

Court reference geometry and homography computation logic are vendored in
the ``_tcd_adapted`` subpackage to avoid pulling in unnecessary
dependencies (the original code requires ``sympy`` and ``matplotlib``
which are not needed for inference).

Model repository : https://github.com/yastrebksv/TennisCourtDetector
Weights download : https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.calibrators._tcd_adapted import (
    CourtReference,
    get_trans_matrix,
    refer_kps,
)
from tennis_virtual_ads.pipeline.calibrators.base import (
    CalibrationResult,
    CourtCalibrator,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 360
NUM_KEYPOINTS = 14
# The model outputs 15 channels (14 court keypoints + 1 centre point used
# only during training).  We only process the first 14.
MODEL_OUTPUT_CHANNELS = 15

# Path to the sibling TennisCourtDetector repository (expected layout:
# capstone-repos/TennisCourtDetector/ alongside capstone-repos/tennis-virtual-ads/).
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_TENNIS_COURT_DETECTOR_REPO = _PROJECT_ROOT.parent / "TennisCourtDetector"


# ---------------------------------------------------------------------------
# Lazy import helper for BallTrackerNet
# ---------------------------------------------------------------------------


def _import_ball_tracker_net() -> type:
    """Import ``BallTrackerNet`` from the sibling TennisCourtDetector repo.

    Uses ``importlib`` to load ``tracknet.py`` directly by file path so
    that we do **not** add the sibling repo to ``sys.path`` (which would
    risk shadowing common module names like ``utils``).

    Raises
    ------
    ImportError
        If ``torch`` is not installed or the sibling repo is not found.
    """
    try:
        import torch  # noqa: F401 — needed by tracknet.py at import time
    except ImportError as exc:
        raise ImportError(
            "TennisCourtDetectorCalibrator requires PyTorch.\n"
            "Install calibration extras:\n"
            "  uv pip install -e '.[calibration]'"
        ) from exc

    tracknet_path = _TENNIS_COURT_DETECTOR_REPO / "tracknet.py"
    if not tracknet_path.exists():
        raise ImportError(
            f"tracknet.py not found at {tracknet_path}.\n"
            "Expected the TennisCourtDetector repo as a sibling directory:\n"
            f"  {_TENNIS_COURT_DETECTOR_REPO}"
        )

    spec = importlib.util.spec_from_file_location("_tcd_tracknet", str(tracknet_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {tracknet_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BallTrackerNet  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Heatmap postprocessing (inlined from TennisCourtDetector/postprocess.py
# to avoid importing sympy which is only needed for refine_kps)
# ---------------------------------------------------------------------------


def _postprocess_heatmap(
    heatmap: np.ndarray,
    scale_x: float = 2.0,
    scale_y: float = 2.0,
    low_threshold: int = 170,
    min_radius: int = 10,
    max_radius: int = 25,
) -> tuple[float | None, float | None]:
    """Extract a single keypoint ``(x, y)`` from a heatmap channel.

    Applies binary thresholding then Hough circle detection to find the
    peak.  The detected centre is scaled from heatmap coordinates to the
    target image coordinates using *scale_x* and *scale_y*.

    Returns ``(None, None)`` when no keypoint is detected.
    """
    _, thresholded = cv2.threshold(heatmap, low_threshold, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        thresholded,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        x_prediction = float(circles[0][0][0] * scale_x)
        y_prediction = float(circles[0][0][1] * scale_y)
        return x_prediction, y_prediction
    return None, None


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class TennisCourtDetectorCalibrator(CourtCalibrator):
    """Court calibrator powered by the TennisCourtDetector neural network.

    Parameters
    ----------
    weights_path : str | Path
        Path to the ``.pt`` weights file for BallTrackerNet.
    device : str | None
        PyTorch device string (``"cpu"``, ``"cuda"``).  Auto-detected
        when ``None``.
    use_homography : bool
        If ``True`` (default), run best-of-12 homography selection after
        keypoint detection to correct shifted keypoints and produce a
        court-to-image homography matrix.
    """

    def __init__(
        self,
        weights_path: str | Path,
        device: str | None = None,
        use_homography: bool = True,
    ) -> None:
        # --- Validate dependencies ----------------------------------------
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "TennisCourtDetectorCalibrator requires PyTorch.\n"
                "Install calibration extras:\n"
                "  uv pip install -e '.[calibration]'"
            ) from exc

        self._torch = torch

        # --- Validate weights file ----------------------------------------
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}.\n"
                "Download from:\n"
                "  https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG\n"
                "and place at:\n"
                f"  {weights_path}"
            )

        # --- Device selection ---------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: str = device

        self._use_homography = use_homography

        # --- Load model ---------------------------------------------------
        BallTrackerNet = _import_ball_tracker_net()
        self._model = BallTrackerNet(out_channels=MODEL_OUTPUT_CHANNELS)
        self._model = self._model.to(self._device)
        self._model.load_state_dict(
            torch.load(str(weights_path), map_location=self._device, weights_only=False)
        )
        self._model.eval()
        logger.info(
            "Loaded TennisCourtDetector model from %s on device=%s",
            weights_path,
            self._device,
        )

        # --- Court reference (for overlay drawing) ------------------------
        self._court_reference = CourtReference()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame: np.ndarray) -> CalibrationResult:
        """Estimate court homography for a single broadcast frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image as ``(H, W, 3)`` BGR array (OpenCV convention).

        Returns
        -------
        CalibrationResult
            ``H`` - 3x3 homography (court-ref to image) or ``None``.
            ``conf`` - fraction of keypoints detected, in ``[0, 1]``.
            ``keypoints`` - ``(14, 2)`` array (``NaN`` for undetected).
            ``debug`` - dict with reprojection error, counts, etc.
        """
        torch = self._torch
        original_height, original_width = frame.shape[:2]

        # Scale factors from model output (640x360) to original image size.
        scale_x = original_width / MODEL_INPUT_WIDTH
        scale_y = original_height / MODEL_INPUT_HEIGHT

        # --- 1. Preprocess: resize + normalise ----------------------------
        resized_frame = cv2.resize(frame, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        normalised = resized_frame.astype(np.float32) / 255.0
        # HWC → CHW, add batch dimension
        input_tensor = torch.tensor(np.rollaxis(normalised, 2, 0)).unsqueeze(0)

        # --- 2. Inference -------------------------------------------------
        with torch.no_grad():
            output = self._model(input_tensor.float().to(self._device))[0]
            prediction = torch.sigmoid(output).detach().cpu().numpy()

        # --- 3. Extract keypoints from heatmaps --------------------------
        raw_keypoints: list[tuple[float | None, float | None]] = []
        for keypoint_index in range(NUM_KEYPOINTS):
            heatmap = (prediction[keypoint_index] * 255).astype(np.uint8)
            x_pred, y_pred = _postprocess_heatmap(heatmap, scale_x=scale_x, scale_y=scale_y)
            raw_keypoints.append((x_pred, y_pred))

        detected_count = sum(1 for kp in raw_keypoints if kp[0] is not None)

        # --- 4. Compute homography ----------------------------------------
        homography_matrix: np.ndarray | None = None
        reprojection_error: float | None = None

        if self._use_homography and detected_count >= 4:
            homography_matrix = get_trans_matrix(raw_keypoints)

            if homography_matrix is not None:
                reprojection_error = self._compute_reprojection_error(
                    raw_keypoints, homography_matrix
                )

        # --- 5. Build (14, 2) keypoints array -----------------------------
        keypoints_array = self._build_keypoints_array(raw_keypoints)

        # --- 6. Confidence ------------------------------------------------
        confidence = detected_count / NUM_KEYPOINTS

        debug: dict[str, Any] = {
            "detected_keypoint_count": detected_count,
            "total_keypoints": NUM_KEYPOINTS,
            "reprojection_error_px": reprojection_error,
            "homography_found": homography_matrix is not None,
            "device": self._device,
            "image_size": (original_width, original_height),
        }

        return CalibrationResult(
            H=homography_matrix,
            conf=confidence,
            keypoints=keypoints_array,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_keypoints_array(
        raw_keypoints: list[tuple[float | None, float | None]],
    ) -> np.ndarray:
        """Convert raw keypoints list to a ``(14, 2)`` float64 array.

        Undetected keypoints are stored as ``(NaN, NaN)``.
        """
        result = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float64)
        for index, (x_value, y_value) in enumerate(raw_keypoints):
            if x_value is not None and y_value is not None:
                result[index] = [x_value, y_value]
        return result

    @staticmethod
    def _compute_reprojection_error(
        raw_keypoints: list[tuple[float | None, float | None]],
        homography: np.ndarray,
    ) -> float | None:
        """Mean reprojection error (pixels) for detected keypoints.

        Projects reference keypoints through *homography* and compares
        with the raw detections.  Returns ``None`` if no comparison
        points are available.
        """
        projected = cv2.perspectiveTransform(refer_kps, homography)
        errors: list[float] = []
        for index in range(NUM_KEYPOINTS):
            if raw_keypoints[index][0] is not None:
                detected = np.array(raw_keypoints[index])
                projected_point = projected[index].flatten()
                errors.append(float(np.linalg.norm(detected - projected_point)))
        if errors:
            return float(np.mean(errors))
        return None

    @property
    def court_reference(self) -> CourtReference:
        """Expose the court reference for overlay drawing."""
        return self._court_reference

    @property
    def court_line_segments(
        self,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """All 10 court line segments as ``((x1,y1), (x2,y2))`` pairs.

        Coordinates are in the court reference system.  Project through
        the homography ``H`` to get image coordinates.
        """
        court_ref = self._court_reference
        return [
            court_ref.baseline_top,
            court_ref.baseline_bottom,
            court_ref.net,
            court_ref.left_court_line,
            court_ref.right_court_line,
            court_ref.left_inner_line,
            court_ref.right_inner_line,
            court_ref.middle_line,
            court_ref.top_inner_line,
            court_ref.bottom_inner_line,
        ]

    @property
    def reference_keypoints(self) -> np.ndarray:
        """Reference keypoints as ``(14, 1, 2)`` array.

        Suitable for ``cv2.perspectiveTransform(reference_keypoints, H)``.
        """
        return refer_kps
