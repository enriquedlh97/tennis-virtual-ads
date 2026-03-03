"""Occlusion maskers -- detect foreground objects that should occlude ads."""

from tennis_virtual_ads.pipeline.maskers.base import OcclusionMasker, OcclusionMaskerResult

# PersonMasker is NOT imported here by default because it requires
# torch + torchvision (heavy optional dependency).  Import it explicitly:
#   from tennis_virtual_ads.pipeline.maskers.person_masker import PersonMasker

# SAM2Masker requires sam2 + torch (heavy optional dependency).  Import explicitly:
#   from tennis_virtual_ads.pipeline.maskers.sam2_masker import SAM2Masker

__all__ = ["OcclusionMasker", "OcclusionMaskerResult"]
