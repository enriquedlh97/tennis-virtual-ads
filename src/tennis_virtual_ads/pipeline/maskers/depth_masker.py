"""Depth-based occlusion masker — Depth Anything V2 (NeurIPS 2024).

Uses monocular depth estimation to separate foreground occluders from
the court plane.  Anything estimated to be *closer* to the camera than
the court surface is treated as a foreground occluder (players, ball,
net, umpire chair, etc.).

This is a novel, class-agnostic approach: instead of detecting specific
object categories, it relies on the geometric prior that the court is
the dominant flat surface at mid-depth, and everything in front of it
should occlude the virtual ad.

Model weights are auto-downloaded from HuggingFace Hub on first run
(~100 MB for Small) and cached in ``~/.cache/huggingface/hub/``.

Dependencies
------------
Requires ``torch`` and ``transformers>=4.40``.  Install via::

    uv sync --extra depth
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)

# HuggingFace model ID templates
_HF_MODEL_IDS: dict[str, str] = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def _check_dependencies() -> None:
    """Raise a clear error if torch or transformers are not installed."""
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "DepthMasker requires PyTorch.  Install the depth extras:\n\n"
            "    uv sync --extra depth\n"
        ) from error
    try:
        import transformers  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "DepthMasker requires HuggingFace transformers>=4.40.  "
            "Install the depth extras:\n\n"
            "    uv sync --extra depth\n"
        ) from error


class DepthMasker(OcclusionMasker):
    """Depth-based occlusion via Depth Anything V2.

    Parameters
    ----------
    model_size : str
        Model variant: ``"small"`` (25M params, fastest), ``"base"`` (98M),
        or ``"large"`` (335M, best quality).
    device : str | None
        PyTorch device string.  Auto-selects CUDA if available.
    court_depth_percentile : float
        Percentile of depth values used to estimate the court plane depth.
        The court is the dominant flat surface — a percentile around 70
        captures it reliably.
    foreground_offset : float
        Depth offset above the court plane threshold.  Pixels with depth
        greater than ``court_depth + foreground_offset`` are considered
        foreground (closer to camera).  Prevents the court surface itself
        from being masked.
    softness : float
        Width of the soft sigmoid transition zone.  ``0`` gives a hard
        binary threshold; positive values produce anti-aliased edges.
    max_internal_size : int
        Maximum input resolution (longest side) for Depth Anything V2.
        Default 518 matches the model's native training resolution.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        court_depth_percentile: float = 70.0,
        foreground_offset: float = 0.05,
        softness: float = 0.02,
        max_internal_size: int = 518,
    ) -> None:
        _check_dependencies()

        import torch

        self._court_depth_percentile = court_depth_percentile
        self._foreground_offset = foreground_offset
        self._softness = softness
        self._max_internal_size = max_internal_size

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        model_id = _HF_MODEL_IDS.get(model_size)
        if model_id is None:
            raise ValueError(
                f"Unknown model_size '{model_size}'. Available: {sorted(_HF_MODEL_IDS.keys())}"
            )

        logger.info("Loading Depth Anything V2 (%s) on %s ...", model_size, self._device)
        load_start = time.perf_counter()

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self._device)
        self._model.eval()

        load_elapsed = time.perf_counter() - load_start
        logger.info("Depth Anything V2 loaded in %.1f s", load_elapsed)

    def mask(self, frame: np.ndarray) -> OcclusionMaskerResult:
        """Generate an occlusion mask from monocular depth estimation.

        Parameters
        ----------
        frame : np.ndarray
            ``(H, W, 3)`` uint8 BGR image (OpenCV convention).

        Returns
        -------
        OcclusionMaskerResult
            ``mask`` is ``(H, W)`` float32 in ``[0, 1]`` — continuous
            alpha where 1.0 = foreground occluder.
        """
        import torch
        from PIL import Image

        frame_h, frame_w = frame.shape[:2]

        # BGR → RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Preprocess and run inference
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # Extract depth map — predicted_depth is (1, H_model, W_model)
        predicted_depth = outputs.predicted_depth

        # Interpolate to original frame size
        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame_h, frame_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        depth_np: np.ndarray = depth.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1] where 1 = closest to camera
        d_min, d_max = float(depth_np.min()), float(depth_np.max())
        if d_max - d_min > 1e-6:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)

        # Estimate court plane depth (court is the dominant flat surface)
        court_depth = float(np.percentile(depth_norm, self._court_depth_percentile))
        threshold = court_depth + self._foreground_offset

        # Apply threshold to produce mask
        if self._softness > 0:
            # Soft sigmoid transition for anti-aliased edges
            mask = 1.0 / (1.0 + np.exp(-(depth_norm - threshold) / self._softness))
        else:
            # Hard binary threshold
            mask = (depth_norm > threshold).astype(np.float32)

        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        return OcclusionMaskerResult(
            mask=mask,
            conf=1.0,
            debug={
                "court_depth": court_depth,
                "threshold": threshold,
                "depth_min": d_min,
                "depth_max": d_max,
                "mask_mean": float(mask.mean()),
                "mask_max": float(mask.max()),
                "mode": "depth",
            },
        )

    def reset_on_cut(self) -> None:
        """Reset on scene cut.  Per-frame model — no temporal state."""
        pass
