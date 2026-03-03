"""Robust Video Matting (RVM) occlusion masker.

Uses the RVM model (PeterL1n/RobustVideoMatting) to produce soft alpha
mattes for people in each frame.  Unlike Mask R-CNN which outputs binary
masks, RVM produces continuous float32 alpha values — a motion-blurred
limb naturally gets alpha ~0.5 instead of a hard edge.

The model's built-in ConvGRU recurrent state provides temporal
consistency across frames (no flicker).  State is reset on scene cuts
via :meth:`reset_on_cut`.

Model weights are auto-downloaded via ``torch.hub`` on first run (~9 MB
for MobileNetv3, ~50 MB for ResNet50) and cached in
``~/.cache/torch/hub/``.

Dependencies
------------
Requires ``torch``.  Install via::

    uv pip install -e ".[matting]"
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Raise a clear error if torch is not installed."""
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "RVMMasker requires PyTorch.  "
            "Install the matting extras:\n\n"
            '    uv pip install -e ".[matting]"\n'
        ) from error


class RVMMasker(OcclusionMasker):
    """Soft alpha matting via Robust Video Matting.

    Parameters
    ----------
    backbone : str
        Model backbone: ``"mobilenetv3"`` (fast, default) or
        ``"resnet50"`` (higher quality, slower).
    downsample_ratio : float
        Internal downsampling ratio for the recurrent decoder.
        ``0.25`` for HD (1080p), ``0.125`` for 4K.
    device : str | None
        PyTorch device string (``"cpu"``, ``"cuda"``, etc.).  When
        ``None`` (default) the masker auto-selects CUDA if available.
    """

    def __init__(
        self,
        backbone: str = "mobilenetv3",
        downsample_ratio: float = 0.25,
        device: str | None = None,
    ) -> None:
        _check_dependencies()

        import torch

        self._downsample_ratio = downsample_ratio

        # Auto-select device.
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info(
            "Loading RVM (backbone=%s) on %s …  "
            "First run will auto-download weights via torch.hub.",
            backbone,
            self._device,
        )
        load_start = time.perf_counter()

        self._model = torch.hub.load(  # type: ignore[no-untyped-call]
            "PeterL1n/RobustVideoMatting",
            backbone,
            trust_repo=True,
        )
        self._model.to(self._device)
        self._model.eval()

        load_elapsed = time.perf_counter() - load_start
        logger.info(
            "RVM loaded in %.1f s (backbone=%s, device=%s, downsample_ratio=%.3f).",
            load_elapsed,
            backbone,
            self._device,
            self._downsample_ratio,
        )

        # ConvGRU recurrent state — 4 tensors, None = uninitialized.
        self._rec: list[Any] = [None, None, None, None]

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
            ``conf`` is ``1.0`` (RVM does not output per-instance scores).
            ``debug`` contains ``alpha_mean`` and ``alpha_max``.
        """
        import torch

        # --- Prepare input tensor ----------------------------------------
        # BGR → RGB, uint8 → float32 [0, 1], reshape to [1, C, H, W].
        rgb = frame[:, :, ::-1].copy()
        src = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        src = src.to(self._device)

        downsample = torch.tensor([self._downsample_ratio], device=self._device)

        # --- Inference ---------------------------------------------------
        with torch.no_grad():
            fgr, pha, *self._rec = self._model(src, *self._rec, downsample)

        # --- Extract alpha -----------------------------------------------
        alpha: np.ndarray = pha[0, 0].cpu().numpy()  # (H, W) float32 [0, 1]

        debug: dict[str, Any] = {
            "alpha_mean": float(alpha.mean()),
            "alpha_max": float(alpha.max()),
        }

        return OcclusionMaskerResult(
            mask=alpha,
            conf=1.0,
            debug=debug,
        )

    def reset_on_cut(self) -> None:
        """Reset ConvGRU recurrent state on scene cut."""
        self._rec = [None, None, None, None]
        logger.debug("RVM recurrent state reset (scene cut).")
