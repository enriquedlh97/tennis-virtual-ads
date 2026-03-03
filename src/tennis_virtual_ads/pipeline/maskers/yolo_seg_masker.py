"""YOLO11-seg occlusion masker — single-model instance segmentation.

Uses a YOLO11 segmentation model (e.g. ``yolo11m-seg.pt``) for fast,
per-frame person masks.  ~5 ms/frame on GPU vs ~50-150 ms for the
SAM2+YOLO detection pipeline.

The model is auto-downloaded by ``ultralytics`` on the first run and
cached locally.

Dependencies
------------
Requires ``ultralytics``.  Install via::

    uv pip install -e ".[yolo]"
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)

# COCO class index for "person" (YOLO uses 0-indexed COCO labels).
_COCO_PERSON_CLASS: int = 0


def _check_dependencies() -> None:
    """Raise a clear error if ultralytics is not installed."""
    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "YOLOSegMasker requires ultralytics.  "
            "Install the yolo extras:\n\n"
            '    uv pip install -e ".[yolo]"\n'
        ) from error


class YOLOSegMasker(OcclusionMasker):
    """Detect people via YOLO11-seg and return a per-frame occlusion mask.

    Parameters
    ----------
    model_name : str
        YOLO segmentation model name (default ``"yolo11m-seg.pt"``).
        Auto-downloaded by ultralytics on first use.
    confidence_threshold : float
        Minimum detection score to include an instance (default ``0.5``).
    device : str | None
        Device string (``"cpu"``, ``"cuda"``, etc.).  When ``None``
        (default) ultralytics auto-selects the best available device.
    """

    def __init__(
        self,
        model_name: str = "yolo11m-seg.pt",
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        _check_dependencies()

        from ultralytics import YOLO

        self._confidence_threshold = confidence_threshold

        logger.info(
            "Loading YOLO-seg model '%s' …  First run will auto-download weights.",
            model_name,
        )
        load_start = time.perf_counter()

        self._model = YOLO(model_name)
        self._device = device

        load_elapsed = time.perf_counter() - load_start
        logger.info("YOLO-seg loaded in %.1f s", load_elapsed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Generate a person occlusion mask for *frame*.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image (OpenCV convention).

        Returns
        -------
        OcclusionMaskerResult
            ``mask`` is ``(H, W)`` float32 in ``[0, 1]``.
        """
        frame_height, frame_width = frame.shape[:2]

        # YOLO expects BGR (same as OpenCV), so no conversion needed.
        results = self._model(
            frame,
            verbose=False,
            classes=[_COCO_PERSON_CLASS],
            conf=self._confidence_threshold,
            device=self._device,
        )

        result = results[0]

        # If no masks were produced, return empty.
        if result.masks is None or len(result.masks.data) == 0:
            return OcclusionMaskerResult(
                mask=np.zeros((frame_height, frame_width), dtype=np.float32),
                conf=0.0,
                debug={"instance_count": 0, "scores": []},
            )

        # result.masks.data is (N, mask_h, mask_w) tensor on model device.
        masks_np = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
        scores = result.boxes.conf.cpu().numpy()  # (N,)

        # Union all instance masks.
        union_mask = np.max(masks_np, axis=0)  # (mask_h, mask_w)

        # Resize to frame resolution (YOLO seg masks may be a different size).
        if union_mask.shape[:2] != (frame_height, frame_width):
            union_mask = cv2.resize(
                union_mask,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR,
            )

        # Binarize at 0.5 for clean mask edges.
        union_mask = (union_mask >= 0.5).astype(np.float32)

        mean_confidence = float(np.mean(scores))

        debug: dict[str, Any] = {
            "instance_count": len(scores),
            "scores": scores.tolist(),
        }

        return OcclusionMaskerResult(
            mask=union_mask,
            conf=mean_confidence,
            debug=debug,
        )

    def reset_on_cut(self) -> None:
        """No-op — YOLO-seg is stateless per-frame."""
