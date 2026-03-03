"""Shared person-detection helpers for occlusion maskers.

Provides YOLO and Mask R-CNN person detection utilities that can be
shared across maskers (currently used by :class:`MatAnyoneMasker`).

Functions
---------
- :func:`yolo_available` — check if ``ultralytics`` is importable.
- :func:`load_yolo` — load a YOLO model.
- :func:`load_mrcnn` — load Mask R-CNN with default COCO weights.
- :func:`detect_persons_yolo` — detect person bounding boxes via YOLO.
- :func:`detect_persons_mrcnn` — detect person bounding boxes via Mask R-CNN.
- :func:`boxes_to_binary_mask` — convert bounding boxes to a filled binary mask.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# COCO label index for "person" in Mask R-CNN predictions.
_COCO_PERSON_LABEL: int = 1


# ------------------------------------------------------------------
# Availability checks
# ------------------------------------------------------------------


def yolo_available() -> bool:
    """Return ``True`` if ``ultralytics`` (YOLO) is importable."""
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------


def load_yolo(model_name: str = "yolo11n.pt") -> Any:
    """Load and return a YOLO model.

    Parameters
    ----------
    model_name : str
        YOLO model filename (auto-downloaded by ultralytics).

    Returns
    -------
    YOLO
        An ``ultralytics.YOLO`` model instance.
    """
    from ultralytics import YOLO  # type: ignore[attr-defined]

    logger.info("Loading YOLO '%s' for person detection ...", model_name)
    model = YOLO(model_name)
    logger.info("YOLO loaded.")
    return model


def load_mrcnn(device: Any) -> tuple[Any, Any]:
    """Load Mask R-CNN with default COCO weights.

    Parameters
    ----------
    device : torch.device
        Target device for the model.

    Returns
    -------
    tuple[model, transforms]
        The Mask R-CNN model (in eval mode) and its associated transforms.
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights,
        maskrcnn_resnet50_fpn,
    )

    logger.info("Loading Mask R-CNN for person detection (YOLO not available) ...")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    transforms = weights.transforms()
    logger.info("Mask R-CNN loaded.")
    return model, transforms


# ------------------------------------------------------------------
# Detection
# ------------------------------------------------------------------


def detect_persons_yolo(
    frame: np.ndarray,
    model: Any,
    threshold: float = 0.5,
) -> tuple[list[list[float]], list[float]]:
    """Detect persons in *frame* using a YOLO model.

    Parameters
    ----------
    frame : np.ndarray
        ``(H, W, 3)`` uint8 BGR image.
    model : YOLO
        An ``ultralytics.YOLO`` model instance.
    threshold : float
        Minimum confidence to keep a detection.

    Returns
    -------
    tuple[list[list[float]], list[float]]
        ``(boxes, scores)`` where each box is ``[x1, y1, x2, y2]``.
    """
    # YOLO expects RGB.
    rgb = frame[:, :, ::-1]
    results = model(rgb, verbose=False)

    boxes: list[list[float]] = []
    scores: list[float] = []

    for result in results:
        for box in result.boxes:
            # COCO class 0 = person.
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= threshold:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                boxes.append(xyxy)
                scores.append(float(box.conf[0]))

    return boxes, scores


def detect_persons_mrcnn(
    frame: np.ndarray,
    model: Any,
    transforms: Any,
    device: Any,
    threshold: float = 0.5,
) -> tuple[list[list[float]], list[float]]:
    """Detect persons in *frame* using Mask R-CNN.

    Parameters
    ----------
    frame : np.ndarray
        ``(H, W, 3)`` uint8 BGR image.
    model : MaskRCNN
        Mask R-CNN model in eval mode.
    transforms : callable
        Preprocessing transforms from the model weights.
    device : torch.device
        Device the model lives on.
    threshold : float
        Minimum confidence to keep a detection.

    Returns
    -------
    tuple[list[list[float]], list[float]]
        ``(boxes, scores)`` where each box is ``[x1, y1, x2, y2]``.
    """
    import torch

    rgb = frame[:, :, ::-1].copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = transforms(tensor)
    tensor = tensor.to(device)

    with torch.inference_mode():
        predictions = model([tensor])[0]

    labels = predictions["labels"].cpu().numpy()
    pred_scores = predictions["scores"].cpu().numpy()
    pred_boxes = predictions["boxes"].cpu().numpy()

    boxes: list[list[float]] = []
    scores: list[float] = []

    for label, score, box in zip(labels, pred_scores, pred_boxes, strict=True):
        if label == _COCO_PERSON_LABEL and score >= threshold:
            boxes.append(box.tolist())
            scores.append(float(score))

    return boxes, scores


# ------------------------------------------------------------------
# Mask generation
# ------------------------------------------------------------------


def boxes_to_binary_mask(
    boxes: list[list[float]],
    height: int,
    width: int,
    padding: int = 10,
) -> np.ndarray:
    """Convert bounding boxes to a filled-rectangle binary mask.

    Parameters
    ----------
    boxes : list[list[float]]
        Bounding boxes as ``[x1, y1, x2, y2]`` (pixel coordinates).
    height, width : int
        Output mask dimensions.
    padding : int
        Pixels to pad each box (clamped to image boundaries).

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 mask with ``255`` inside any box, ``0`` elsewhere.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x1 = max(0, int(box[0]) - padding)
        y1 = max(0, int(box[1]) - padding)
        x2 = min(width, int(box[2]) + padding)
        y2 = min(height, int(box[3]) + padding)
        mask[y1:y2, x1:x2] = 255
    return mask
