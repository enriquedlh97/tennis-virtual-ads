"""MatAnyone occlusion masker — best-quality video matting (CVPR 2025).

Uses the MatAnyone model for high-quality alpha matting with temporal
consistency via memory-bank propagation.  Produces continuous float32
alpha mattes (not binary masks), yielding smooth edges on
motion-blurred limbs and fine hair/racket details.

The masker auto-detects persons on the first frame (and optionally
every ``reprompt_interval`` frames) using YOLO or Mask R-CNN, converts
detections to a binary prompt mask, and lets MatAnyone propagate
through subsequent frames via its memory bank.

Model weights are auto-downloaded from HuggingFace Hub on first run
(~350 MB) and cached in ``~/.cache/huggingface/hub/``.

Dependencies
------------
Requires ``matanyone`` and ``torch``.  Install via::

    pip install git+https://github.com/pq-yang/MatAnyone.git
    uv sync --all-extras
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Raise a clear error if matanyone or torch are not installed."""
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "MatAnyoneMasker requires PyTorch.  "
            "Install the matanyone extras:\n\n"
            "    pip install git+https://github.com/pq-yang/MatAnyone.git\n"
        ) from error
    try:
        import matanyone  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "MatAnyoneMasker requires the matanyone package.  "
            "Install it with:\n\n"
            "    pip install git+https://github.com/pq-yang/MatAnyone.git\n"
        ) from error


class MatAnyoneMasker(OcclusionMasker):
    """High-quality alpha matting via MatAnyone (CVPR 2025).

    Parameters
    ----------
    device : str | None
        PyTorch device string.  Auto-selects CUDA if available.
    confidence_threshold : float
        Minimum detection score for person detections.
    reprompt_interval : int
        Re-run person detection every N frames.  ``0`` means never
        re-prompt — rely entirely on MatAnyone's temporal propagation.
    n_warmup : int
        Number of warmup repetitions on the first frame to stabilize
        the memory bank before processing subsequent frames.
    use_yolo : bool
        Use YOLO for person detection (preferred).  Falls back to
        Mask R-CNN if YOLO is unavailable or ``use_yolo=False``.
    yolo_model : str
        YOLO model filename (auto-downloaded by ultralytics).
    checkpoint_path : str | None
        Path to a local MatAnyone ``.pth`` checkpoint.  If ``None``,
        auto-downloads from HuggingFace Hub (``PeiqingYang/MatAnyone``).
    max_internal_size : int
        Maximum internal processing resolution (longest side).
        ``-1`` means no limit (use original resolution).
    box_padding : int
        Pixel padding added to each detection box when constructing
        the binary prompt mask.
    """

    def __init__(
        self,
        device: str | None = None,
        confidence_threshold: float = 0.5,
        reprompt_interval: int = 0,
        n_warmup: int = 5,
        use_yolo: bool = True,
        yolo_model: str = "yolo11n.pt",
        checkpoint_path: str | None = None,
        max_internal_size: int = -1,
        box_padding: int = 10,
    ) -> None:
        _check_dependencies()

        import torch

        self._confidence_threshold = confidence_threshold
        self._reprompt_interval = reprompt_interval
        self._n_warmup = n_warmup
        self._max_internal_size = max_internal_size
        self._box_padding = box_padding

        # Auto-select device.
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # --- Load MatAnyone --------------------------------------------------
        logger.info(
            "Loading MatAnyone on %s (checkpoint=%s) ...",
            self._device,
            checkpoint_path or "PeiqingYang/MatAnyone (HuggingFace Hub)",
        )
        load_start = time.perf_counter()

        from matanyone import InferenceCore

        if checkpoint_path is not None:
            # Load from local checkpoint via the MatAnyone model class.
            from matanyone.model.matanyone import MatAnyone as MatAnyoneModel

            model = MatAnyoneModel.from_pretrained(checkpoint_path)
            self._processor = InferenceCore(model, device=self._device)
        else:
            # Auto-download from HuggingFace Hub.
            self._processor = InferenceCore("PeiqingYang/MatAnyone", device=self._device)

        load_elapsed = time.perf_counter() - load_start
        logger.info("MatAnyone loaded in %.1f s", load_elapsed)

        # --- Load person detector --------------------------------------------
        from tennis_virtual_ads.pipeline.maskers._person_detector import (
            load_mrcnn,
            load_yolo,
            yolo_available,
        )

        self._use_yolo = use_yolo and yolo_available()
        self._yolo_model_obj: Any = None
        self._mrcnn_model: Any = None
        self._mrcnn_transforms: Any = None

        if self._use_yolo:
            self._yolo_model_obj = load_yolo(yolo_model)
        else:
            self._mrcnn_model, self._mrcnn_transforms = load_mrcnn(self._device)

        # --- State -----------------------------------------------------------
        self._is_initialized: bool = False
        self._frame_count: int = 0
        self._last_prompt_frame: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Generate a soft alpha matte for people in *frame*.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image (OpenCV convention).

        Returns
        -------
        OcclusionMaskerResult
            ``mask`` is ``(H, W)`` float32 in ``[0, 1]`` — continuous
            alpha (not binarized).
        """
        self._frame_count += 1

        # Decide whether to (re-)prompt.
        need_prompt = not self._is_initialized
        if (
            self._reprompt_interval > 0
            and self._is_initialized
            and (self._frame_count - self._last_prompt_frame) >= self._reprompt_interval
        ):
            need_prompt = True

        if need_prompt:
            return self._prompt_and_warmup(frame)
        else:
            return self._propagate(frame)

    def reset_on_cut(self) -> None:
        """Reset MatAnyone memory on scene cut (forces re-prompting)."""
        self._processor.clear_memory()
        self._is_initialized = False
        logger.debug("MatAnyone memory cleared (scene cut).")

    # ------------------------------------------------------------------
    # Internal — prompt mode
    # ------------------------------------------------------------------

    def _prompt_and_warmup(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Detect persons, initialize MatAnyone, and run warmup."""
        import torch

        frame_h, frame_w = frame.shape[:2]

        # --- Detect persons --------------------------------------------------
        boxes, scores = self._detect_persons(frame)
        if len(boxes) == 0:
            self._is_initialized = False
            return OcclusionMaskerResult(
                mask=np.zeros((frame_h, frame_w), dtype=np.float32),
                conf=0.0,
                debug={"instance_count": 0, "scores": [], "mode": "prompt_empty"},
            )

        # --- Build binary prompt mask ----------------------------------------
        from tennis_virtual_ads.pipeline.maskers._person_detector import (
            boxes_to_binary_mask,
        )

        binary_mask = boxes_to_binary_mask(boxes, frame_h, frame_w, padding=self._box_padding)

        # --- Prepare tensors -------------------------------------------------
        image_tensor = self._frame_to_tensor(frame)
        mask_tensor = torch.from_numpy(binary_mask).float().to(self._device)
        # Normalize mask to [0, 1] (it's currently 0/255).
        mask_tensor = mask_tensor / 255.0

        # Optionally resize for VRAM savings.
        image_tensor, mask_tensor, scale = self._maybe_resize(image_tensor, mask_tensor)

        # --- Initialize MatAnyone --------------------------------------------
        self._processor.clear_memory()

        # First call: provide mask + objects to initialize.
        with torch.inference_mode():
            output_prob = self._processor.step(image_tensor, mask_tensor, objects=[1])
            # Second call: first_frame_pred to seed memory.
            output_prob = self._processor.step(image_tensor, first_frame_pred=True)

            # Warmup: repeat first frame to stabilize memory bank.
            for _ in range(self._n_warmup):
                output_prob = self._processor.step(image_tensor, first_frame_pred=True)

        # --- Extract alpha ---------------------------------------------------
        alpha = self._extract_alpha(output_prob, frame_h, frame_w, scale)

        self._is_initialized = True
        self._last_prompt_frame = self._frame_count

        mean_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0
        return OcclusionMaskerResult(
            mask=alpha,
            conf=mean_conf,
            debug={
                "instance_count": len(boxes),
                "scores": scores,
                "mode": "prompt",
                "n_warmup": self._n_warmup,
                "alpha_mean": float(alpha.mean()),
                "alpha_max": float(alpha.max()),
            },
        )

    # ------------------------------------------------------------------
    # Internal — propagation mode
    # ------------------------------------------------------------------

    def _propagate(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Propagate mask to *frame* via MatAnyone's memory bank."""
        import torch

        frame_h, frame_w = frame.shape[:2]

        image_tensor = self._frame_to_tensor(frame)
        image_tensor, _, scale = self._maybe_resize(image_tensor, None)

        with torch.inference_mode():
            output_prob = self._processor.step(image_tensor)

        alpha = self._extract_alpha(output_prob, frame_h, frame_w, scale)

        return OcclusionMaskerResult(
            mask=alpha,
            conf=1.0,
            debug={
                "instance_count": -1,
                "scores": [],
                "mode": "propagate",
                "alpha_mean": float(alpha.mean()),
                "alpha_max": float(alpha.max()),
            },
        )

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _detect_persons(self, frame: np.ndarray) -> tuple[list[list[float]], list[float]]:
        """Detect persons using YOLO or Mask R-CNN."""
        from tennis_virtual_ads.pipeline.maskers._person_detector import (
            detect_persons_mrcnn,
            detect_persons_yolo,
        )

        if self._use_yolo and self._yolo_model_obj is not None:
            return detect_persons_yolo(frame, self._yolo_model_obj, self._confidence_threshold)
        elif self._mrcnn_model is not None:
            return detect_persons_mrcnn(
                frame,
                self._mrcnn_model,
                self._mrcnn_transforms,
                self._device,
                self._confidence_threshold,
            )
        return [], []

    def _frame_to_tensor(self, frame: np.ndarray) -> Any:
        """Convert a BGR uint8 frame to an RGB float32 ``(3, H, W)`` tensor."""
        import torch

        rgb = frame[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.to(self._device)

    def _maybe_resize(
        self,
        image: Any,
        mask: Any | None,
    ) -> tuple[Any, Any | None, float]:
        """Optionally downscale if ``max_internal_size`` is set.

        Returns ``(image, mask, scale)`` where *scale* is the ratio of
        original to resized dimensions (``1.0`` if no resize).
        """
        if self._max_internal_size <= 0:
            return image, mask, 1.0

        import torch.nn.functional as F

        _, h, w = image.shape
        longest = max(h, w)
        if longest <= self._max_internal_size:
            return image, mask, 1.0

        scale = self._max_internal_size / longest
        new_h = int(h * scale)
        new_w = int(w * scale)
        # Make divisible by 16 (MatAnyone requirement).
        new_h = (new_h + 15) // 16 * 16
        new_w = (new_w + 15) // 16 * 16

        image = F.interpolate(
            image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        )[0]

        if mask is not None:
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode="nearest"
            )[0, 0]

        return image, mask, scale

    def _extract_alpha(
        self,
        output_prob: Any,
        orig_h: int,
        orig_w: int,
        scale: float,
    ) -> np.ndarray:
        """Extract ``(H, W)`` float32 alpha from MatAnyone output probability.

        Handles variable output shapes and resizes back to original
        dimensions if internal resolution was reduced.
        """
        alpha_tensor = self._processor.output_prob_to_mask(output_prob, matting=True)
        alpha: np.ndarray = np.asarray(alpha_tensor.cpu().numpy(), dtype=np.float32)

        # Squeeze to (H, W) regardless of output shape.
        alpha = np.squeeze(alpha)

        # Clamp to [0, 1].
        alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

        # Resize back to original if we downscaled.
        if scale != 1.0 or alpha.shape != (orig_h, orig_w):
            alpha = np.asarray(
                cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR),
                dtype=np.float32,
            )

        return alpha
