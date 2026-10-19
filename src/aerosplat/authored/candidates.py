"""Pipeline proposals -> `candidate` access points, without overruling a person (ACCESS_POINTS §3.4).

Whatever proposes an opening -- the periodic-grid fit today, a detector later (§5.4) --
goes through here, so the rules are the same for all of them:

* a proposal only ever becomes a `candidate` with `origin: pipeline`; never `verified`
  (ADR-0009): a person confirms it in the facade editor;
* a proposal that lands on something already decided is dropped: a `verified` feature
  is there already, and a `rejected` one is the memory that stops the pipeline proposing
  the same wrong thing again (§3.4). An existing `candidate` or `stale` feature drops it
  too, so running the same proposer twice adds nothing.

"Lands on" is checked in 3-D, not by facade id: a feature remembers the facade frame
it was drawn in (`host`), and a recomputed run may have renamed or shifted that facade
(§5.9). Projecting the stored shape into the current facade's frame catches the same
wall under a new name; a feature on another wall (another normal, or well off the plane)
never blocks anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from aerosplat.authored.geometry import make_access_point, rect_uv
from aerosplat.authored.models import AccessPoint, Lifecycle, Origin
from aerosplat.semantics.models import Facade

Rect = tuple[float, float, float, float]          # u0, u1, v0, v1 in a facade's frame

# A proposal is dropped when this share of the smaller of the two shapes overlaps.
# The smaller, not the union: a hand-drawn window pair contains two proposed panes,
# and each of them is mostly inside it while their IoU with it is under a half.
MIN_OVERLAP = 0.3
# Two shapes are on the same wall when their normals are this close ...
SAME_WALL_DEG = 20.0
# ... and the stored shape's centre is this near the current plane (a fitted plane
# moves by up to its own scatter between runs).
SAME_PLANE_M = 1.5


@dataclass(frozen=True)
class Proposal:
    """One shape a proposer wants a person to look at."""

    rect: Rect
    confidence: float
    notes: str


def overlap(a: Rect, b: Rect) -> float:
    """Intersection over the smaller of the two areas, 0 when they do not touch."""
    du = min(a[1], b[1]) - max(a[0], b[0])
    dv = min(a[3], b[3]) - max(a[2], b[2])
    if du <= 0 or dv <= 0:
        return 0.0
    smaller = min((a[1] - a[0]) * (a[3] - a[2]), (b[1] - b[0]) * (b[3] - b[2]))
    return du * dv / smaller if smaller > 0 else 0.0


def rect_on(facade: Facade, feature: AccessPoint, *, plane_m: float = SAME_PLANE_M) -> Optional[Rect]:
    """The feature's shape as a (u, v) box in `facade`'s frame, or None if it is elsewhere.

    The stored polygon is in the frame it was drawn in; its corners go to ENU through
    that frame and come back through the current one. A pad, or a shape on a wall
    facing another way, is None.
    """
    h, g = feature.host, feature.geometry
    if h is None or g is None:
        return None
    n_cur = np.asarray(facade.normal, float)
    if float(np.dot(np.asarray(h.normal, float), n_cur)) < math.cos(math.radians(SAME_WALL_DEG)):
        return None
    origin = np.asarray(h.origin, float)
    u_axis = np.asarray(h.u_axis, float)
    corners = np.array([origin + u * u_axis + np.array([0.0, 0.0, v - origin[2]])
                        for u, v in g.polygon_uv])
    centre = corners.mean(axis=0)
    if abs(float(facade.distance_to_plane(centre))) > plane_m:
        return None
    us, vs = facade.uv_of(corners)
    return float(us.min()), float(us.max()), float(vs.min()), float(vs.max())


def screen(proposals: Sequence[Proposal], facade: Facade, existing: Iterable[AccessPoint], *,
           plane_m: Optional[float] = None) -> tuple[list[Proposal], dict[str, int]]:
    """Drop the proposals that land on an existing feature; count what dropped them.

    Returns the proposals to create and `{lifecycle: dropped}`. Every lifecycle blocks:
    verified and rejected because a person decided, candidate and stale because a
    person is about to.
    """
    tol = plane_m if plane_m is not None else max(SAME_PLANE_M, 2.0 * facade.plane_rms_m)
    taken: list[tuple[Rect, Lifecycle]] = []
    for f in existing:
        r = rect_on(facade, f, plane_m=tol)
        if r is not None:
            taken.append((r, f.lifecycle))
    keep: list[Proposal] = []
    dropped: dict[str, int] = {}
    for p in proposals:
        hit = next((lc for r, lc in taken if overlap(p.rect, r) >= MIN_OVERLAP), None)
        if hit is None:
            keep.append(p)
        else:
            dropped[hit.value] = dropped.get(hit.value, 0) + 1
    return keep, dropped


@dataclass(frozen=True)
class PadProposal:
    """A pad zone a proposer (aerosplat.semantics.pads) wants a person to look at."""

    class_id: str                                   # pad.roof_flat | pad.ground
    polygon_enu: list[tuple[float, float]]
    z: float
    building_id: Optional[str]                      # the roof's building; None on the ground
    confidence: float
    notes: str
    evidence: list[str]


# Two pads are the same place when their outlines overlap this much (of the smaller) and
# their surfaces are this close in height (a roof pad never blocks the yard below it).
SAME_PAD_DZ_M = 2.0


def screen_pads(proposals: Sequence[PadProposal], existing: Iterable[AccessPoint]
                ) -> tuple[list[PadProposal], dict[str, int]]:
    """`screen` for pads: drop proposals on an existing pad of any lifecycle (§3.4)."""
    from aerosplat.semantics.pads import polygon_overlap

    taken = [(f.pad, f.lifecycle) for f in existing if f.pad is not None]
    keep: list[PadProposal] = []
    dropped: dict[str, int] = {}
    for p in proposals:
        hit = next((lc for pad, lc in taken
                    if abs(pad.z - p.z) < SAME_PAD_DZ_M
                    and polygon_overlap(p.polygon_enu, pad.polygon_enu) >= MIN_OVERLAP), None)
        if hit is None:
            keep.append(p)
        else:
            dropped[hit.value] = dropped.get(hit.value, 0) + 1
    return keep, dropped


def make_pad_candidates(proposals: Sequence[PadProposal], *, site_key: str, taxonomy: str,
                        model: str, created_by: str, run_id: Optional[str],
                        epoch_id: Optional[str]) -> list[AccessPoint]:
    """`candidate` pad records, never verified (ADR-0009). Surface and hazards are unknown."""
    from aerosplat.authored.geometry import make_pad_point

    out: list[AccessPoint] = []
    for p in proposals:
        ap = make_pad_point(
            site_key=site_key, class_id=p.class_id, taxonomy=taxonomy,
            polygon_enu=p.polygon_enu, z=p.z, building_id=p.building_id,
            attributes={"surface": "unknown", "hazard": "unknown"},
            lifecycle=Lifecycle.candidate, origin=Origin.pipeline, created_by=created_by,
            run_id=run_id, epoch_id=epoch_id, evidence=list(p.evidence), notes=p.notes,
            confidence=float(min(0.99, max(0.01, p.confidence))))
        ap.provenance.model = model
        out.append(ap)
    return out


def make_candidates(proposals: Sequence[Proposal], *, site_key: str, facade: Facade,
                    class_id: str, taxonomy: str, model: str, created_by: str,
                    run_id: Optional[str], epoch_id: Optional[str],
                    evidence: Sequence[str]) -> list[AccessPoint]:
    """`candidate` records for the proposals, with the proposer named in the provenance."""
    out: list[AccessPoint] = []
    for p in proposals:
        u0, u1, v0, v1 = p.rect
        ap = make_access_point(
            site_key=site_key, facade=facade, class_id=class_id, taxonomy=taxonomy,
            polygon_uv=rect_uv(u0, v0, u1, v1), lifecycle=Lifecycle.candidate,
            origin=Origin.pipeline, created_by=created_by, run_id=run_id, epoch_id=epoch_id,
            evidence=list(evidence), notes=p.notes,
            confidence=float(min(0.99, max(0.01, p.confidence))))
        ap.provenance.model = model
        out.append(ap)
    return out
