"""Occlusion maskers -- detect foreground objects that should occlude ads."""

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

# PersonMasker is NOT imported here by default because it requires
# torch + torchvision (heavy optional dependency).  Import it explicitly:
#   from tennis_virtual_ads.pipeline.maskers.person_masker import PersonMasker

# SAM2Masker requires sam2 + torch (heavy optional dependency).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.sam2_masker import SAM2Masker

# YOLOSegMasker requires ultralytics (optional dependency).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.yolo_seg_masker import YOLOSegMasker

# ColorKeyMasker is pure OpenCV (no extra deps).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.color_key_masker import ColorKeyMasker

# RVMMasker requires torch (heavy optional dependency).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.rvm_masker import RVMMasker

# MatAnyoneMasker requires matanyone + torch (heavy optional dependency).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.matanyone_masker import MatAnyoneMasker

# DepthMasker requires torch + transformers (heavy optional dependency).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.depth_masker import DepthMasker

# MatAnyoneHybridMasker requires matanyone + torch (heavy optional dependency).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.matanyone_hybrid_masker import MatAnyoneHybridMasker

# SAM2VideoMasker requires sam2 + torch (heavy optional dependency, offline only).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.sam2_video_masker import SAM2VideoMasker

__all__ = ["OcclusionMasker", "OcclusionMaskerResult"]
