"""Where the occupancy grid is built (the `occupancy_extent` rules).

Outside the grid every query answers OUTSIDE, which blocks like occupied space, so the extent
decides which missions can be planned at all. It must therefore be a stated choice, never a
side effect of how far the cloud happens to reach: an Earth Studio capture of one building
reconstructs ~1.6 × 1.8 km around it, and a grid over all of it is billions of voxels.

In order of precedence (`extent="auto"`):

1. an explicit box (`occupancy_bbox`, Site ENU metres);
2. the Site's bbox (WGS84, `site.yaml`), mapped to Site ENU;
3. a square of ±`radius_m` around the Site origin — the centre of the capture.

`extent="radius"` takes (3) even when the Site has a bbox; `extent="raster"` the whole terrain
raster. Whatever is chosen is logged and recorded in occupancy.json (`GridExtent`). The grid is
never cropped to fit `max_voxels`: a choice that does not fit fails the stage with a hint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.surface.models import GridExtent, parse_bbox

EXTENT_MODES = ("auto", "radius", "raster")
Box = tuple[float, float, float, float]


def parse_site_bbox(text: str):
    """"west,south,east,north" in degrees -> store.models.BBox (validated)."""
    from aerosplat.store.models import BBox

    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError(f"site bbox must be 'west,south,east,north', got {text!r}")
    try:
        west, south, east, north = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"site bbox must hold four numbers, got {text!r}") from None
    return BBox(west=west, south=south, east=east, north=north)


def site_bbox_enu(bbox, origin: GeoPoint, samples_per_edge: int = 16) -> Box:
    """The axis-aligned Site ENU box that holds a WGS84 bbox.

    The edges are sampled, not just the corners: meridians converge, so in ENU a WGS84 box is
    a slightly curved quadrilateral, and the envelope must hold all of it.
    """
    t = np.linspace(0.0, 1.0, samples_per_edge + 1)
    w, s, e, n = bbox.west, bbox.south, bbox.east, bbox.north
    lat = np.concatenate([np.full_like(t, s), np.full_like(t, n), s + t * (n - s), s + t * (n - s)])
    lon = np.concatenate([w + t * (e - w), w + t * (e - w), np.full_like(t, w), np.full_like(t, e)])
    enu = EnuFrame(origin).from_geodetic(lat, lon, np.full_like(lat, origin.h))
    # Between samples an edge bulges by micrometres (km-sized boxes): 1 cm covers it.
    pad = 0.01
    return (float(enu[:, 0].min()) - pad, float(enu[:, 1].min()) - pad,
            float(enu[:, 0].max()) + pad, float(enu[:, 1].max()) + pad)


def _fmt(box: Box) -> str:
    return f"x {box[0]:.1f}…{box[2]:.1f}, y {box[1]:.1f}…{box[3]:.1f} м"


@dataclass(frozen=True)
class ExtentChoice:
    source: str                 # "bbox" | "site_bbox" | "radius" | "raster"
    box: Optional[Box]          # the requested box in Site ENU; None: the whole raster
    why: str                    # one line for the log

    def report(self, raster: Box, grid: Box, res_m: float) -> GridExtent:
        """What was asked and what the grid became: the raster may end before the box does."""
        clipped: list[str] = []
        if self.box is not None:
            for side, want, got, sign in (("захід", self.box[0], grid[0], 1),
                                          ("південь", self.box[1], grid[1], 1),
                                          ("схід", self.box[2], grid[2], -1),
                                          ("північ", self.box[3], grid[3], -1)):
                if sign * (got - want) > res_m:
                    clipped.append(side)
        return GridExtent(source=self.source, requested=self.box, raster=raster, grid=grid,
                          clipped=clipped, detail=self.why)


def choose_extent(*, extent: str = "auto", bbox: Optional[str] = None, radius_m: float = 150.0,
                  site_bbox=None, origin: Optional[GeoPoint] = None) -> ExtentChoice:
    """Apply the rules of the module docstring. Contradictory or incomplete input fails."""
    if extent not in EXTENT_MODES:
        raise ValueError(f"extent must be one of {EXTENT_MODES}, got {extent!r}")
    if not (math.isfinite(radius_m) and radius_m > 0):
        raise ValueError(f"radius must be positive, got {radius_m}")
    if bbox is not None and extent != "auto":
        raise ValueError(f"an explicit bbox and extent={extent!r} contradict each other")
    if site_bbox is not None and origin is None:
        raise ValueError("the Site bbox needs the Site origin to be mapped to Site ENU")
    square = (-radius_m, -radius_m, radius_m, radius_m)
    if extent == "raster":
        return ExtentChoice("raster", None, "увесь растр рельєфу (occupancy_extent=raster)")
    if extent == "radius":
        return ExtentChoice("radius", square, f"квадрат ±{radius_m:g} м навколо початку Site "
                                              f"(occupancy_extent=radius)")
    if bbox is not None:
        box = parse_bbox(bbox)
        return ExtentChoice("bbox", box, f"occupancy_bbox з конфігурації: {_fmt(box)}")
    if site_bbox is not None:
        assert origin is not None
        box = site_bbox_enu(site_bbox, origin)
        return ExtentChoice("site_bbox", box,
                            f"bbox ділянки (W {site_bbox.west:.6f}, S {site_bbox.south:.6f}, "
                            f"E {site_bbox.east:.6f}, N {site_bbox.north:.6f}) → {_fmt(box)}")
    return ExtentChoice("radius", square, f"квадрат ±{radius_m:g} м навколо початку Site "
                                          f"(у ділянки немає bbox; occupancy_radius)")
