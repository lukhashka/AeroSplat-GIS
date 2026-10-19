"""Authored features across runs: which ones belong to a facade, and where the others land.

A feature is drawn on one run's facade and keeps that facade's frame (`HostFacade`). Two
questions follow when there is more than one run (ACCESS_POINTS §5.9):

* **Is it native to this facade?** Facade ids are per run, so `host.facade_id` alone is not
  enough: `B04/SW1` of one run can be a different building in the next. A feature is native
  to a facade when it carries exactly that facade's frame (`is_native`).
* **Where does a feature of another run land in this run?** Its shape goes to ENU through
  the frame it was drawn in and is tested against every facade of this run with the same
  rules as facade matching (`aerosplat.semantics.matching`): normals, plane offset within
  the planes' scatter, and the shape inside the facade's extent (`place_feature`).

The result is `placed` (exactly one facade), `ambiguous` (several facades could hold it,
e.g. two parallel fits of one wall) or `none`. Only `placed` may be carried over, and only
by an explicit request (`reanchor`): nothing here writes, and a feature is never moved
because a new run appeared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence

import numpy as np

from aerosplat.authored.geometry import geometry_from_uv, host_of
from aerosplat.authored.models import AccessPoint
from aerosplat.semantics.matching import MatchParams
from aerosplat.semantics.models import Facade

PlacementStatus = Literal["placed", "ambiguous", "none", "not_on_facade"]

# Frames copied from a facade round-trip through JSON exactly; this only absorbs noise.
SAME_FRAME_TOL = 1e-6
# The shape must lie this much inside a facade's extent (by area), its centre within the
# extent grown by `EXTENT_MARGIN_M`: a window drawn at the very edge of a wall that the
# other run fitted a little shorter is still that wall.
MIN_INSIDE = 0.5
EXTENT_MARGIN_M = 0.5


def is_native(feature: AccessPoint, facade: Facade, tol: float = SAME_FRAME_TOL) -> bool:
    """True when `feature` was drawn on exactly this facade (same id and same frame)."""
    h = feature.host
    if h is None or h.facade_id != facade.id:
        return False
    return (np.allclose(h.origin, facade.origin, atol=tol, rtol=0)
            and np.allclose(h.normal, facade.normal, atol=tol, rtol=0)
            and np.allclose(h.u_axis, facade.u_axis, atol=tol, rtol=0)
            and abs(h.v_ground - facade.v_ground) <= tol)


def native_features(features: Iterable[AccessPoint], facade: Facade) -> list[AccessPoint]:
    return [f for f in features if is_native(f, facade)]


def corners_enu(feature: AccessPoint) -> Optional[np.ndarray]:
    """The polygon's corners in Site ENU, through the frame it was drawn in."""
    h, g = feature.host, feature.geometry
    if h is None or g is None:
        return None
    origin = np.asarray(h.origin, float)
    u_axis = np.asarray(h.u_axis, float)
    return np.array([origin + u * u_axis + np.array([0.0, 0.0, v - origin[2]])
                     for u, v in g.polygon_uv])


@dataclass(frozen=True)
class FacadeFit:
    """How a feature lies against one facade."""

    facade_id: str
    angle_deg: float
    offset_m: float                       # distance of the shape's centre from the plane
    offset_tol_m: float
    inside: float                         # share of the shape's (u, v) box inside the facade
    polygon_uv: list[tuple[float, float]]  # the shape in this facade's frame


@dataclass
class Placement:
    feature_id: str
    status: PlacementStatus
    facade_id: Optional[str] = None       # for `placed`
    fit: Optional[FacadeFit] = None       # for `placed`
    candidates: list[FacadeFit] = field(default_factory=list)
    reason: str = ""


def fit_on(feature: AccessPoint, facade: Facade,
           params: MatchParams = MatchParams()) -> Optional[FacadeFit]:
    """The feature's shape in `facade`'s frame, or None when it is not on that wall."""
    corners = corners_enu(feature)
    if corners is None:
        return None
    assert feature.host is not None
    n_host = np.asarray(feature.host.normal, float)
    angle = math.degrees(math.acos(float(np.clip(np.dot(n_host, np.asarray(facade.normal, float)),
                                                 -1.0, 1.0))))
    if angle > params.max_angle_deg:
        return None
    centre = corners.mean(axis=0)
    # The drawn-on plane's scatter is not on the record, so it counts as this one's.
    tol = params.offset_tol(facade.plane_rms_m, facade.plane_rms_m)
    offset = abs(float(facade.distance_to_plane(centre)))
    if offset > tol:
        return None
    us, vs = facade.uv_of(corners)
    u0, u1, v0, v1 = float(us.min()), float(us.max()), float(vs.min()), float(vs.max())
    uc, vc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    m = EXTENT_MARGIN_M
    if not (facade.u_min - m <= uc <= facade.u_max + m and facade.v_ground - m <= vc <= facade.v_max + m):
        return None
    du = max(0.0, min(u1, facade.u_max + m) - max(u0, facade.u_min - m))
    dv = max(0.0, min(v1, facade.v_max + m) - max(v0, facade.v_ground - m))
    area = max((u1 - u0) * (v1 - v0), 1e-9)
    inside = du * dv / area
    if inside < MIN_INSIDE:
        return None
    return FacadeFit(facade_id=facade.id, angle_deg=angle, offset_m=offset, offset_tol_m=tol,
                     inside=inside, polygon_uv=[(float(u), float(v)) for u, v in zip(us, vs)])


def place_feature(feature: AccessPoint, facades: Sequence[Facade],
                  params: MatchParams = MatchParams()) -> Placement:
    """Which facade of a run (if exactly one) the feature lies on."""
    if feature.host is None or feature.geometry is None:
        return Placement(feature_id=feature.id, status="not_on_facade",
                         reason="pad: не на фасаді")
    fits = [f for f in (fit_on(feature, fc, params) for fc in facades) if f is not None]
    fits.sort(key=lambda f: (f.offset_m, f.facade_id))
    if not fits:
        return Placement(feature_id=feature.id, status="none",
                         reason=f"жоден фасад цього run-а не містить її (кут ≤ "
                                f"{params.max_angle_deg:g}°, зсув у межах розкиду площини, "
                                f"≥ {100 * MIN_INSIDE:.0f}% форми в межах фасаду)")
    if len(fits) > 1:
        return Placement(feature_id=feature.id, status="ambiguous", candidates=fits,
                         reason="її можуть містити кілька фасадів: "
                                + ", ".join(f"{f.facade_id} (зсув {f.offset_m:.2f} м)" for f in fits)
                                + " — не вибираємо")
    f = fits[0]
    return Placement(feature_id=feature.id, status="placed", facade_id=f.facade_id, fit=f,
                     candidates=fits,
                     reason=f"{f.facade_id}: кут {f.angle_deg:.1f}°, зсув {f.offset_m:.2f} м")


def foreign_on(facade: Facade, run_facades: Sequence[Facade], features: Iterable[AccessPoint],
               params: MatchParams = MatchParams()) -> list[Placement]:
    """Features of other runs that land on `facade`, placed against all of its run's facades.

    "Of other runs" = native to none of `run_facades`. A feature that is ambiguous between
    this facade and another is included, so the editor can say why it cannot be carried.
    """
    out = []
    for f in features:
        if f.host is None or any(is_native(f, fc) for fc in run_facades):
            continue
        if fit_on(f, facade, params) is None:
            continue
        out.append(place_feature(f, run_facades, params))
    return out


def reanchor(feature: AccessPoint, facade: Facade, run_facades: Sequence[Facade],
             params: MatchParams = MatchParams()) -> tuple[AccessPoint, Placement]:
    """The feature re-expressed in `facade`'s frame, if it is `placed` exactly there.

    Raises ValueError otherwise. The shape is the drawn corners projected onto the new
    plane: v (height) is unchanged, u is measured along the new wall, and the plane offset
    (`fit.offset_m`) is what the feature moves by. Lifecycle and history are kept. The
    address is dropped: it names the old facade and its rows, so it is recomputed on the new
    one (and a confirmed target on this feature sees `target_changed` until re-confirmed,
    because its fingerprint covers the host and the geometry). The caller stores the result
    with an audit entry.
    """
    placement = place_feature(feature, run_facades, params)
    if placement.status != "placed" or placement.facade_id != facade.id:
        raise ValueError(f"{feature.id} не можна перенести на {facade.id}: {placement.reason}")
    assert placement.fit is not None
    geometry = geometry_from_uv(facade, placement.fit.polygon_uv)
    moved = feature.model_copy(update={"host": host_of(facade), "geometry": geometry,
                                       "address": None}, deep=True)
    return moved, placement
