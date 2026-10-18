"""Turning a polygon drawn on a facade orthophoto into a stored access point.

The editor works in facade (u, v) metres, which is all the operator sees. Everything
else on the record — the ENU centre, the outward normal, the metric size — follows
from that polygon and the facade frame it was drawn in, and is filled in here so a
consumer never has to redo the conversion (or get the u-axis convention wrong).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from aerosplat.authored.models import (AccessPoint, Geometry, HostFacade, Lifecycle, Origin,
                                       PadGeometry, Provenance, new_ap_id)
from aerosplat.semantics.models import Facade

# Slivers are almost always a stray click rather than a real opening.
MIN_SIDE_M = 0.05
# A pad smaller than this cannot hold a drone, whatever the class says.
MIN_PAD_AREA_M2 = 1.0


def host_of(facade: Facade) -> HostFacade:
    return HostFacade(facade_id=facade.id, building_id=facade.building_id, origin=facade.origin,
                      normal=facade.normal, u_axis=facade.u_axis, v_ground=facade.v_ground,
                      azimuth_deg=facade.azimuth_deg)


def geometry_from_uv(facade: Facade, polygon_uv: Sequence[tuple[float, float]]) -> Geometry:
    """Derive the §3.1 geometry block from a polygon in the facade's (u, v) frame."""
    if len(polygon_uv) < 3:
        raise ValueError("a polygon needs at least three points")
    us = [float(p[0]) for p in polygon_uv]
    vs = [float(p[1]) for p in polygon_uv]
    width, height = max(us) - min(us), max(vs) - min(vs)
    if width < MIN_SIDE_M or height < MIN_SIDE_M:
        raise ValueError(f"the shape is {width:.3f} × {height:.3f} m: too thin to be an opening "
                         f"(both sides must exceed {MIN_SIDE_M * 100:.0f} cm)")
    u_c, v_c = (min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2
    centre = facade.point(u_c, v_c)
    return Geometry(polygon_uv=[(u, v) for u, v in zip(us, vs)],
                    center=(float(centre[0]), float(centre[1]), float(centre[2])),
                    normal=facade.normal, u_axis=facade.u_axis,
                    width=width, height=height, u_center=u_c, v_center=v_c)


def pad_geometry(polygon_enu: Sequence[tuple[float, float]], z: float,
                 building_id: Optional[str] = None) -> PadGeometry:
    """A flat pad outline (east, north) at height z -> the stored block, with its area centroid."""
    pts = [(float(p[0]), float(p[1])) for p in polygon_enu]
    if len(pts) < 3:
        raise ValueError("a pad outline needs at least three points")
    if not all(math.isfinite(c) for p in pts for c in p) or not math.isfinite(z):
        raise ValueError("a pad outline must be finite")
    # Shoelace: signed area and area centroid of a simple polygon.
    a2 = cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        cross = x0 * y1 - x1 * y0
        a2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area = abs(a2) / 2
    if area < MIN_PAD_AREA_M2:
        raise ValueError(f"the pad is {area:.2f} m²: too small to land on "
                         f"(at least {MIN_PAD_AREA_M2:g} m²)")
    return PadGeometry(polygon_enu=pts, z=float(z), building_id=building_id,
                       center=(cx / (3 * a2), cy / (3 * a2), float(z)), area_m2=area)


def rect_uv(u0: float, v0: float, u1: float, v1: float) -> list[tuple[float, float]]:
    """A rectangle in (u, v), counter-clockwise from its lower-left corner."""
    lo_u, hi_u = min(u0, u1), max(u0, u1)
    lo_v, hi_v = min(v0, v1), max(v0, v1)
    return [(lo_u, lo_v), (hi_u, lo_v), (hi_u, hi_v), (lo_u, hi_v)]


def make_access_point(*, site_key: str, facade: Facade, class_id: str, taxonomy: str,
                      polygon_uv: Sequence[tuple[float, float]],
                      attributes: Optional[dict[str, str]] = None,
                      lifecycle: Lifecycle = Lifecycle.verified,
                      origin: Origin = Origin.manual, created_by: str = "op:unknown",
                      run_id: Optional[str] = None, epoch_id: Optional[str] = None,
                      evidence: Optional[list[str]] = None, notes: str = "",
                      confidence: Optional[float] = None,
                      feature_id: Optional[str] = None) -> AccessPoint:
    """Assemble a complete record. The class is not checked here — the taxonomy does that."""
    return AccessPoint(
        id=feature_id or new_ap_id(), site_key=site_key, **{"class": class_id},
        kind=class_id.split(".", 1)[0], taxonomy=taxonomy, host=host_of(facade),
        geometry=geometry_from_uv(facade, polygon_uv), attributes=dict(attributes or {}),
        lifecycle=lifecycle, confidence=confidence, notes=notes,
        provenance=Provenance(origin=origin, run_id=run_id, epoch_id=epoch_id,
                              created_by=created_by, evidence=list(evidence or [])))


def make_pad_point(*, site_key: str, class_id: str, taxonomy: str,
                   polygon_enu: Sequence[tuple[float, float]], z: float,
                   building_id: Optional[str] = None,
                   attributes: Optional[dict[str, str]] = None,
                   lifecycle: Lifecycle = Lifecycle.verified,
                   origin: Origin = Origin.manual, created_by: str = "op:unknown",
                   run_id: Optional[str] = None, epoch_id: Optional[str] = None,
                   evidence: Optional[list[str]] = None, notes: str = "",
                   feature_id: Optional[str] = None) -> AccessPoint:
    """A pad record (§3.2): a flat outline in Site ENU instead of a facade polygon."""
    return AccessPoint(
        id=feature_id or new_ap_id(), site_key=site_key, **{"class": class_id},
        kind=class_id.split(".", 1)[0], taxonomy=taxonomy,
        pad=pad_geometry(polygon_enu, z, building_id), attributes=dict(attributes or {}),
        lifecycle=lifecycle, notes=notes,
        provenance=Provenance(origin=origin, run_id=run_id, epoch_id=epoch_id,
                              created_by=created_by, evidence=list(evidence or [])))
