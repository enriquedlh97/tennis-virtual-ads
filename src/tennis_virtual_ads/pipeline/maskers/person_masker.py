"""Person occlusion masker using torchvision Mask R-CNN.

Detects people (players, ball kids, umpires) in each frame and returns
a union mask of all person instances.  The mask is ``1.0`` where a
person is present and ``0.0`` elsewhere.

The model (``maskrcnn_resnet50_fpn`` with COCO-pretrained weights) is
loaded lazily on first instantiation.  Weights are auto-downloaded by
torchvision on the first run (~170 MB) and cached locally.

Dependencies
------------
Requires ``torch`` and ``torchvision``.  Install via::

    uv pip install -e ".[occlusion]"
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)

# COCO label index for "person".
_COCO_PERSON_LABEL: int = 1


def _check_dependencies() -> None:
    """Raise a clear error if torch / torchvision are not installed."""
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "PersonMasker requires PyTorch.  "
            "Install the occlusion extras:\n\n"
            '    uv pip install -e ".[occlusion]"\n'
        ) from error

    try:
        import torchvision  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "PersonMasker requires torchvision.  "
            "Install the occlusion extras:\n\n"
            '    uv pip install -e ".[occlusion]"\n'
        ) from error


class PersonMasker(OcclusionMasker):
    """Detect people via Mask R-CNN and return a per-frame occlusion mask.

    Parameters
    ----------
    confidence_threshold : float
        Minimum detection score to include an instance (default ``0.5``).
    device : str | None
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).  When
        ``None`` (default) the masker auto-selects CUDA if available.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        _check_dependencies()

        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )

        self._confidence_threshold = confidence_threshold

        # Auto-select device.
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info(
            "Loading Mask R-CNN (maskrcnn_resnet50_fpn) on %s …  "
            "First run will auto-download weights (~170 MB).",
            self._device,
        )
        load_start = time.perf_counter()

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self._model = maskrcnn_resnet50_fpn(weights=weights)
        self._model.to(self._device)
        self._model.eval()

        # Store the transform that the weights expect (normalisation etc.).
        self._transforms = weights.transforms()

        load_elapsed = time.perf_counter() - load_start
        logger.info("Mask R-CNN loaded in %.1f s (device=%s).", load_elapsed, self._device)

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
            ``conf`` is the mean detection score of accepted instances
            (or ``0.0`` if no person was found).
            ``debug`` contains ``instance_count`` and ``scores``.
        """
        import torch

        frame_height, frame_width = frame.shape[:2]

        # --- Prepare input tensor ----------------------------------------
        # BGR → RGB, uint8 → float32 [0, 1], then apply model transforms.
        rgb_frame = frame[:, :, ::-1].copy()  # BGR → RGB (contiguous copy)
        tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        tensor_frame = self._transforms(tensor_frame)
        tensor_frame = tensor_frame.to(self._device)

        # --- Inference ---------------------------------------------------
        with torch.no_grad():
            predictions = self._model([tensor_frame])[0]

        # --- Filter for person class with sufficient confidence ----------
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        masks = predictions["masks"]  # (N, 1, H, W) float32 in [0, 1]

        is_person = labels == _COCO_PERSON_LABEL
        is_confident = scores >= self._confidence_threshold
        keep = is_person & is_confident

        kept_scores = scores[keep]
        kept_masks = masks[keep]  # (K, 1, H, W)

        # --- Merge instance masks into a single union mask ---------------
        if len(kept_masks) == 0:
            union_mask = np.zeros((frame_height, frame_width), dtype=np.float32)
            mean_confidence = 0.0
        else:
            # Convert to numpy; take max across instances (union).
            kept_masks_np = kept_masks[:, 0, :, :].cpu().numpy()  # (K, H, W)
            union_mask = np.max(kept_masks_np, axis=0)  # (H, W) in [0, 1]
            # Binarize at 0.5 for clean mask edges (Mask R-CNN outputs soft
            # probabilities per instance).
            union_mask = (union_mask >= 0.5).astype(np.float32)
            mean_confidence = float(np.mean(kept_scores))

        debug: dict[str, Any] = {
            "instance_count": int(kept_scores.shape[0]),
            "scores": kept_scores.tolist(),
        }

        return OcclusionMaskerResult(
            mask=union_mask,
            conf=mean_confidence,
            debug=debug,
        )
