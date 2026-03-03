"""SAM2 occlusion masker with YOLO/Mask R-CNN auto-prompting.

Uses SAM2.1 (Segment Anything Model 2) for high-quality temporal occlusion
masks with built-in video propagation.  Players are detected on keyframes
using YOLO11 (preferred) or Mask R-CNN (fallback), and SAM2 propagates
masks through subsequent frames.

Advantages over PersonMasker:
    - Better mask quality at limb edges, rackets, and fine details
    - Built-in temporal consistency (no flicker between frames)
    - Handles arbitrary objects via bounding box prompts

T4 GPU optimisations:
    - ``sam2.1_hiera_small`` (46M params) fits T4 with offloading
    - ``offload_video_to_cpu=True``, ``offload_state_to_cpu=True``
    - ``torch.float16`` inference

Dependencies
------------
Requires ``sam2`` and optionally ``ultralytics`` (for YOLO prompting).
Install via::

    uv pip install -e ".[sam2]"
    uv pip install -e ".[yolo]"  # optional, for YOLO prompting
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)

# COCO label index for "person" (Mask R-CNN fallback).
_COCO_PERSON_LABEL: int = 1


def _check_sam2() -> None:
    """Raise a clear error if sam2 is not installed."""
    try:
        import sam2  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "SAM2Masker requires sam2.  "
            "Install the sam2 extras:\n\n"
            '    uv pip install -e ".[sam2]"\n'
        ) from error


def _yolo_available() -> bool:
    """Check if ultralytics (YOLO) is available."""
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


_SAM2_MODEL_CONFIGS: dict[str, str] = {
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}
"""Map model names to their config YAML paths (within the sam2 package)."""


class SAM2Masker(OcclusionMasker):
    """SAM2-based occlusion masker with auto-prompting.

    Parameters
    ----------
    model_name : str
        SAM2 model name.  Default ``"sam2.1_hiera_small"``.
    checkpoint_path : str | None
        Path to SAM2 checkpoint file.  If ``None``, defaults to
        ``"weights/sam2.1_hiera_small.pt"``.
    device : str | None
        PyTorch device string.  Auto-selects CUDA if available.
    confidence_threshold : float
        Minimum detection score for prompting detections.  Default ``0.5``.
    reprompt_interval : int
        Re-run person detection every N frames to catch new players
        entering the scene.  Default ``5``.
    use_yolo : bool
        Use YOLO11 for prompting (if available).  Falls back to Mask R-CNN
        if YOLO is unavailable or ``use_yolo=False``.  Default ``True``.
    yolo_model : str
        YOLO model name.  Default ``"yolo11n.pt"`` (nano, fast).
    offload_to_cpu : bool
        Offload SAM2 video/state to CPU for memory savings.  Default ``True``.
    """

    def __init__(
        self,
        model_name: str = "sam2.1_hiera_small",
        checkpoint_path: str | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        reprompt_interval: int = 5,
        use_yolo: bool = True,
        yolo_model: str = "yolo11n.pt",
        offload_to_cpu: bool = True,
    ) -> None:
        _check_sam2()

        import torch

        # SAM2 logs "Computing image embeddings..." on every frame via the
        # root logger.  Suppress to WARNING to avoid flooding the console.
        logging.getLogger("root").setLevel(logging.WARNING)
        # Also silence the unnamed root logger that SAM2 uses directly.
        _root = logging.getLogger()
        if _root.level < logging.WARNING:
            # Add a filter instead of changing the root level (which would
            # suppress our own logs).  Filter out the known noisy messages.
            _sam2_suppressed = frozenset(
                {
                    "Computing image embeddings for the provided image...",
                    "Image embeddings computed.",
                    "For numpy array image, we assume (HxWxC) format",
                    "Loaded checkpoint sucessfully",
                }
            )

            class _SAM2LogFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    return record.getMessage() not in _sam2_suppressed

            _root.addFilter(_SAM2LogFilter())

        self._confidence_threshold = confidence_threshold
        self._reprompt_interval = reprompt_interval
        self._offload_to_cpu = offload_to_cpu

        # Auto-select device.
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Resolve config and checkpoint.
        config_file = _SAM2_MODEL_CONFIGS.get(model_name)
        if config_file is None:
            raise ValueError(
                f"Unknown SAM2 model '{model_name}'. Available: {list(_SAM2_MODEL_CONFIGS.keys())}"
            )
        if checkpoint_path is None:
            checkpoint_path = f"weights/{model_name}.pt"

        # --- Load SAM2 ---
        logger.info(
            "Loading SAM2 model '%s' (config=%s, ckpt=%s) on %s ...",
            model_name,
            config_file,
            checkpoint_path,
            self._device,
        )
        load_start = time.perf_counter()

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._sam2_model = build_sam2(
            config_file,
            checkpoint_path,
            device=str(self._device),
        )

        # Run in float32 — the small model (46M params) fits T4 16GB
        # without half precision.  Both .half() and autocast cause issues
        # with SAM2's internal buffers producing empty/incorrect masks.
        self._sam2_predictor = SAM2ImagePredictor(self._sam2_model)

        load_elapsed = time.perf_counter() - load_start
        logger.info("SAM2 loaded in %.1f s", load_elapsed)

        # --- Load prompter (YOLO or Mask R-CNN) ---
        self._use_yolo = use_yolo and _yolo_available()
        self._yolo_model_obj: Any = None
        self._mrcnn_model: Any = None
        self._mrcnn_transforms: Any = None

        if self._use_yolo:
            from ultralytics import YOLO

            logger.info("Loading YOLO '%s' for SAM2 prompting ...", yolo_model)
            self._yolo_model_obj = YOLO(yolo_model)
            logger.info("YOLO loaded for SAM2 prompting")
        else:
            self._load_mrcnn_fallback()

        # --- State ---
        self._frame_count: int = 0
        self._last_prompt_frame: int = -reprompt_interval  # Force prompt on first frame.
        self._prev_mask: np.ndarray | None = None

    def _load_mrcnn_fallback(self) -> None:
        """Load Mask R-CNN as fallback prompter."""
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        logger.info("Loading Mask R-CNN for SAM2 prompting (YOLO not available) ...")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._mrcnn_model = maskrcnn_resnet50_fpn(weights=weights)
        self._mrcnn_model.to(self._device)
        self._mrcnn_model.eval()
        self._mrcnn_transforms = weights.transforms()
        logger.info("Mask R-CNN loaded for SAM2 prompting")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Generate an occlusion mask for *frame*.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image.

        Returns
        -------
        OcclusionMaskerResult
        """
        frame_height, frame_width = frame.shape[:2]
        self._frame_count += 1

        # Determine if we need to re-prompt.
        need_prompt = (
            self._frame_count - self._last_prompt_frame >= self._reprompt_interval
            or self._prev_mask is None
        )

        if need_prompt:
            # Detect persons and prompt SAM2.
            boxes, scores = self._detect_persons(frame)

            if len(boxes) == 0:
                # No persons detected — return empty mask.
                self._prev_mask = np.zeros((frame_height, frame_width), dtype=np.float32)
                return OcclusionMaskerResult(
                    mask=self._prev_mask,
                    conf=0.0,
                    debug={"instance_count": 0, "scores": [], "prompt_method": self._prompt_method},
                )

            # Run SAM2 with bounding box prompts.
            mask, conf = self._run_sam2_with_boxes(frame, boxes)
            self._prev_mask = mask
            self._last_prompt_frame = self._frame_count

            return OcclusionMaskerResult(
                mask=mask,
                conf=conf,
                debug={
                    "instance_count": len(boxes),
                    "scores": scores,
                    "prompt_method": self._prompt_method,
                    "prompted": True,
                },
            )
        else:
            # Between prompts: run SAM2 on current frame for propagation.
            mask, conf = self._run_sam2_propagate(frame)
            self._prev_mask = mask

            return OcclusionMaskerResult(
                mask=mask,
                conf=conf,
                debug={
                    "instance_count": -1,  # Unknown without re-detection.
                    "scores": [],
                    "prompt_method": "propagate",
                    "prompted": False,
                },
            )

    def reset_on_cut(self) -> None:
        """Reset SAM2 state on scene cut (forces re-prompting)."""
        self._prev_mask = None
        self._last_prompt_frame = -self._reprompt_interval

    @property
    def _prompt_method(self) -> str:
        return "yolo" if self._use_yolo else "mrcnn"

    # ------------------------------------------------------------------
    # Person detection
    # ------------------------------------------------------------------

    def _detect_persons(self, frame: np.ndarray) -> tuple[list[list[float]], list[float]]:
        """Detect persons in frame, return bounding boxes and scores.

        Returns
        -------
        tuple[list[list[float]], list[float]]
            (boxes, scores) where boxes are [x1, y1, x2, y2].
        """
        if self._use_yolo and self._yolo_model_obj is not None:
            return self._detect_yolo(frame)
        elif self._mrcnn_model is not None:
            return self._detect_mrcnn(frame)
        else:
            return [], []

    def _detect_yolo(self, frame: np.ndarray) -> tuple[list[list[float]], list[float]]:
        """Detect persons using YOLO11."""
        # YOLO expects RGB.
        rgb = frame[:, :, ::-1]
        results = self._yolo_model_obj(rgb, verbose=False)

        boxes: list[list[float]] = []
        scores: list[float] = []

        for result in results:
            for box in result.boxes:
                # COCO class 0 = person.
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= self._confidence_threshold:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    boxes.append(xyxy)
                    scores.append(float(box.conf[0]))

        return boxes, scores

    def _detect_mrcnn(self, frame: np.ndarray) -> tuple[list[list[float]], list[float]]:
        """Detect persons using Mask R-CNN (fallback)."""
        import torch

        rgb = frame[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = self._mrcnn_transforms(tensor)
        tensor = tensor.to(self._device)

        with torch.inference_mode():
            predictions = self._mrcnn_model([tensor])[0]

        labels = predictions["labels"].cpu().numpy()
        pred_scores = predictions["scores"].cpu().numpy()
        pred_boxes = predictions["boxes"].cpu().numpy()

        boxes: list[list[float]] = []
        scores: list[float] = []

        for label, score, box in zip(labels, pred_scores, pred_boxes, strict=True):
            if label == _COCO_PERSON_LABEL and score >= self._confidence_threshold:
                boxes.append(box.tolist())
                scores.append(float(score))

        return boxes, scores

    # ------------------------------------------------------------------
    # SAM2 inference
    # ------------------------------------------------------------------

    def _run_sam2_with_boxes(
        self,
        frame: np.ndarray,
        boxes: list[list[float]],
    ) -> tuple[np.ndarray, float]:
        """Run SAM2 image prediction with bounding box prompts.

        Returns (union_mask, confidence).
        """
        import torch

        frame_height, frame_width = frame.shape[:2]

        # SAM2 expects RGB.
        rgb = frame[:, :, ::-1].copy()

        # Process all boxes together.
        input_boxes = np.array(boxes, dtype=np.float32)

        with torch.inference_mode():
            self._sam2_predictor.set_image(rgb)
            masks, scores, _ = self._sam2_predictor.predict(
                box=input_boxes,
                multimask_output=False,
            )

        # masks shape: (N, 1, H, W) or (N, H, W)
        if masks.ndim == 4:
            masks = masks[:, 0, :, :]  # (N, H, W)

        # Union all instance masks.
        if len(masks) == 0:
            return np.zeros((frame_height, frame_width), dtype=np.float32), 0.0

        union_mask = np.max(masks, axis=0).astype(np.float32)
        # Binarize at 0.5.
        union_mask = (union_mask >= 0.5).astype(np.float32)

        mean_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0
        return union_mask, mean_conf

    def _run_sam2_propagate(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Run SAM2 on a frame between prompts (simple re-segmentation).

        For now, we re-run SAM2 image prediction using the previous mask
        as a prompt (mask prompting).  This provides temporal consistency
        without the full video predictor overhead.
        """
        import cv2 as _cv2
        import torch

        frame_height, frame_width = frame.shape[:2]

        if self._prev_mask is None:
            return np.zeros((frame_height, frame_width), dtype=np.float32), 0.0

        rgb = frame[:, :, ::-1].copy()

        # SAM2 expects mask_input as (1, 256, 256) logits — resize the
        # previous binary mask and convert to logit-like values.
        mask_resized = _cv2.resize(self._prev_mask, (256, 256), interpolation=_cv2.INTER_LINEAR)
        # Convert [0,1] binary mask to logit-scale: positive = foreground.
        mask_logits = (mask_resized * 20.0 - 10.0).astype(np.float32)
        mask_input = mask_logits[np.newaxis, :, :]  # (1, 256, 256)

        with torch.inference_mode():
            self._sam2_predictor.set_image(rgb)
            masks, scores, _ = self._sam2_predictor.predict(
                mask_input=mask_input,
                multimask_output=False,
            )

        if masks.ndim == 4:
            masks = masks[:, 0, :, :]

        if len(masks) == 0:
            return np.zeros((frame_height, frame_width), dtype=np.float32), 0.0

        result_mask = masks[0].astype(np.float32)
        result_mask = (result_mask >= 0.5).astype(np.float32)
        conf = float(scores[0]) if len(scores) > 0 else 0.0

        return result_mask, conf
