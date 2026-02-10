"""Ad placement -- warp an RGBA ad image onto the court surface."""

from tennis_virtual_ads.pipeline.placer.ad_placer import AdPlacer
from tennis_virtual_ads.pipeline.placer.placement import PlacementSpec, compute_ad_court_corners

__all__ = ["AdPlacer", "PlacementSpec", "compute_ad_court_corners"]
