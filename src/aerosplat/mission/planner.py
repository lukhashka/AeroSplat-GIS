"""Approach geometry and feasibility of one mission (ACCESS_POINTS §7.2, SIMULATION §3.1).

From a confirmed target and the occupancy grid of the twin:

    opening:  H = c + a·standoff + up·Δh,  E = c + a·D_entry + up·Δh,  heading = −a
    pad:      H = c + up·standoff,          E = H + a·L  at the transit height
    corridor: r = r_drone + m + k·σ_total,  σ_total² = σ_AP² + σ_georef² + σ_GNSS²

`a` is the target's normal `n` unless that corridor is blocked: then directions inside the
class's approach cone (half its apex angle each side) are tried and the clearest is taken.
Directions whose E or H lie too close to the grid's edge are not tried at all.
Every number of the target's class — standoff, m, cone, D_entry, Δh, the σ gate — comes from
the taxonomy through the target's `HandoverSpec`; the vehicle's numbers from the profile.

The path is home → straight up to the transit height → over E → down to E → H (hold) and
back the same way, then land at home. Every leg is checked against the grid with its own
radius: σ_AP enters only the legs near the target, transit uses σ_georef ⊕ σ_GNSS. The
result is `feasible`, `feasible_with_risks` or `infeasible`, with named reasons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from aerosplat.authored.models import AccessPoint, new_ulid
from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.mission.corridor import (SQRT3, LegCheck, Plane, check_leg, column_ceiling,
                                        findings_of, sample_path, surface_z)
from aerosplat.mission.export import geodetic
from aerosplat.mission.models import (MISSION_PREFIX, AltitudeFrame, Approach, Code, Fence,
                                      Finding, HomePoint, Leg, MissionItem, MissionPlan,
                                      MissionRefused, MissionTarget, OccupancyRef,
                                      PlanningProfile, Severity, SigmaUsed, Vec3,
                                      require_relative, verdict_of)
from aerosplat.mission.targets import planning_findings
from aerosplat.surface.models import FREE, ColumnSource
from aerosplat.surface.occupancy import OccupancyGrid

F64 = NDArray[np.float64]
UP = np.array([0.0, 0.0, 1.0])

# MAVLink commands used by v0 (common.xml).
NAV_WAYPOINT, NAV_LAND, NAV_TAKEOFF = 16, 21, 22
CONDITION_YAW, DO_CHANGE_SPEED = 115, 178
# Vertices of the polygon that stands in for each disc of the fence (circumscribed).
FENCE_DISC_VERTICES = 16


def _v(p) -> Vec3:
    a = np.asarray(p, float)
    return (float(a[0]), float(a[1]), float(a[2]))


def _unit(v) -> F64:
    a = np.asarray(v, float)
    n = float(np.linalg.norm(a))
    if not (math.isfinite(n) and n > 1e-9):
        raise ValueError(f"not a direction: {v}")
    return a / n


def heading_deg(towards) -> float:
    """Compass yaw (0 = north, 90 = east) of a horizontal direction."""
    d = np.asarray(towards, float)
    return math.degrees(math.atan2(float(d[0]), float(d[1]))) % 360.0


def cone_directions(n, half_angle_deg: float, tilt_steps: int,
                    azimuths: int) -> list[tuple[F64, float]]:
    """`n` first, then directions tilted up to `half_angle_deg` from it, ring by ring."""
    n = _unit(n)
    out = [(n, 0.0)]
    if half_angle_deg <= 0 or tilt_steps <= 0:
        return out
    ref = UP if abs(float(n @ UP)) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(n, ref))
    e2 = np.cross(n, e1)
    for s in range(1, tilt_steps + 1):
        theta = math.radians(half_angle_deg * s / tilt_steps)
        for k in range(azimuths):
            phi = 2 * math.pi * k / azimuths
            a = math.cos(theta) * n + math.sin(theta) * (math.cos(phi) * e1 + math.sin(phi) * e2)
            out.append((_unit(a), math.degrees(theta)))
    return out


def convex_hull(points: F64) -> F64:
    """Andrew's monotone chain; counter-clockwise, no repeated first vertex."""
    pts = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[F64] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[F64] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def fence_polygon(centres: F64, radii: F64) -> F64:
    """Convex polygon holding a disc of `radii[i]` around every `centres[i]` (x, y).

    Each disc is replaced by its circumscribed regular polygon, so the hull contains the discs,
    and the hull of discs at a segment's ends contains the segment buffered by the smaller.
    """
    k = FENCE_DISC_VERTICES
    ang = 2 * math.pi * (np.arange(k) + 0.5) / k
    ring = np.column_stack([np.cos(ang), np.sin(ang)]) / math.cos(math.pi / k)
    pts = (centres[:, None, :2] + radii[:, None, None] * ring[None]).reshape(-1, 2)
    return convex_hull(pts)


def _finding(code: Code, detail: str, *, leg: Optional[str] = None, at=None,
             severity: Severity = Severity.blocking) -> Finding:
    return Finding(code=code, severity=severity, leg=leg, detail=detail,
                   at=_v(at) if at is not None else None)


# ── the grid's extent ────────────────────────────────────────────────────
def grid_margin_m(grid: OccupancyGrid, xy) -> float:
    """Horizontal distance from (x, y) to the nearest lateral edge of the grid; < 0 outside."""
    s = grid.spec
    x, y = float(xy[0]), float(xy[1])
    return min(x - s.x0, s.x1 - x, y - s.y0, s.y1 - y)


def grid_shortfalls(grid: OccupancyGrid,
                    points: list[tuple[str, F64, float]]) -> list[tuple[str, F64, float, float]]:
    """(name, point, needed, has) of every point closer to the grid's edge than its radius
    plus two voxels: the column search of the transit height looks that much further."""
    short = []
    for name, p, radius in points:
        need = radius + 2.0 * grid.res_m
        have = grid_margin_m(grid, p)
        if have < need:
            short.append((name, np.asarray(p, float), need, have))
    return short


def grid_extent(grid: OccupancyGrid) -> str:
    s = grid.spec
    return f"x {s.x0:.1f}…{s.x1:.1f}, y {s.y0:.1f}…{s.y1:.1f} m"


def rebuild_hint(grid: OccupancyGrid, short: list[tuple[str, F64, float, float]]) -> str:
    """How to rebuild occupancy so that the points `grid_shortfalls` found fit: the grid's
    extent grown to hold each with its need plus 10 m, rounded out to 10 m."""
    s = grid.spec
    pad = 10.0
    xs = [s.x0, s.x1] + [v for _, p, need, _ in short for v in (p[0] - need - pad, p[0] + need + pad)]
    ys = [s.y0, s.y1] + [v for _, p, need, _ in short for v in (p[1] - need - pad, p[1] + need + pad)]
    box = (math.floor(min(xs) / 10) * 10, math.floor(min(ys) / 10) * 10,
           math.ceil(max(xs) / 10) * 10, math.ceil(max(ys) / 10) * 10)
    return (f"Rebuild occupancy with an extent that holds it, e.g. --set occupancy_bbox="
            f"{box[0]:g},{box[1]:g},{box[2]:g},{box[3]:g} (or a larger occupancy_radius, or a "
            f"Site bbox)")


def require_inside_grid(grid: OccupancyGrid, points: list[tuple[str, F64, float]], *,
                        note: str = "") -> None:
    """Refuse a mission whose points, with the corridor radius around each, leave the grid.

    Beyond the grid everything is OUTSIDE and blocks, so such a mission is not "obstructed":
    it was never checked, and the remedy is a bigger grid, not another route. The grid is a
    rectangle and legs are straight, so checking each leg's ends checks the whole corridor.
    `note` follows the list of points in the refusal.
    """
    short = grid_shortfalls(grid, points)
    if not short:
        return
    what = "; ".join(f"{name} ({p[0]:.1f}, {p[1]:.1f}) needs {need:.1f} m to the edge, has "
                     f"{have:.1f}" for name, p, need, have in short)
    raise MissionRefused([_finding(
        Code.outside_grid,
        f"the mission leaves the occupancy grid ({grid_extent(grid)}): {what}{note}. Outside the "
        f"grid nothing is known, so the corridor cannot be checked. {rebuild_hint(grid, short)}",
        at=short[0][1])])


def _entry_hover(H: F64, E: F64, radius: float) -> list[tuple[str, F64, float]]:
    return [("entry E", E, radius), ("hover H", H, radius)]


# A direction the cone search may choose: (a, tilt from n in degrees, H, E).
Direction = tuple[F64, float, F64, F64]


def directions_inside_grid(grid: OccupancyGrid, directions: list[Direction], radius: float,
                           cone_deg: float) -> list[Direction]:
    """The directions of the cone (the direct approach first) whose E and H keep `radius` plus
    two voxels from the grid's edge, in their order; refuse when there is none.

    Only these are tried: the plan checks E and H of the chosen direction with
    `require_inside_grid`, so a direction that falls short would be refused even when another
    one fits, and its corridor, reaching beyond the grid, cannot be checked anyway (a wide one
    took minutes to find that out). When none fits, the refusal is certain, so it is given
    before any corridor is checked; it names the direct approach.
    """
    fit = [d for d in directions if not grid_shortfalls(grid, _entry_hover(d[2], d[3], radius))]
    if not fit:
        _, _, H, E = directions[0]
        require_inside_grid(grid, _entry_hover(H, E, radius), note="" if len(directions) == 1 else (
            f" (all {len(directions) - 1} other directions of the {cone_deg:g}° approach cone "
            f"fall short too, so no corridor was checked)"))
    return fit


# ── the approach ─────────────────────────────────────────────────────────
def corridor_radii(target: MissionTarget, prof: PlanningProfile) -> tuple[SigmaUsed, float, float]:
    """(σ used, r_app, r_tr): the corridor radius near the target and in transit (§7.2)."""
    # σ of the position source: GNSS, or the declared ExternalNav of a GNSS-free profile (ADR-0015).
    s_ap, s_geo, s_gnss = target.sigma.ap.value_m, target.sigma.georef.value_m, prof.sigma_nav_m
    sigma = SigmaUsed(ap_m=s_ap, georef_m=s_geo, gnss_m=s_gnss, nav_source=prof.nav_source,
                      target_m=math.hypot(s_ap, s_geo),
                      approach_m=math.sqrt(s_ap ** 2 + s_geo ** 2 + s_gnss ** 2),
                      transit_m=math.hypot(s_geo, s_gnss))
    r_app = prof.r_drone_m + target.handover.min_clearance_m + prof.k_sigma * sigma.approach_m
    r_tr = prof.r_drone_m + prof.transit_margin_m + prof.k_sigma * sigma.transit_m
    return sigma, r_app, r_tr


@dataclass(frozen=True)
class TargetGeometry:
    kind: str                       # "opening" | "pad"
    c: F64
    n: F64                          # outward for an opening, up for a pad
    dh: F64                         # up·Δh (an opening's hover offset; 0 for a pad)
    d_entry: float                  # D_entry (0 for a pad)
    plane: Plane                    # the target surface, which the approach does not count


def target_geometry(target: MissionTarget, ap: AccessPoint, relief_m: float) -> TargetGeometry:
    spec = target.handover
    if ap.geometry is not None:                                   # an opening on a facade
        c = np.asarray(ap.geometry.center, float)
        n = _unit(ap.geometry.normal)
        assert spec.entry_distance_m is not None                 # resolve_handover guarantees it
        kind, dh, d_entry = "opening", UP * spec.hover_dh_m, spec.entry_distance_m
    else:
        assert ap.pad is not None
        c = np.asarray(ap.pad.center, float)
        kind, n, dh, d_entry = "pad", UP.copy(), np.zeros(3), 0.0
    return TargetGeometry(kind=kind, c=c, n=n, dh=dh, d_entry=d_entry,
                          plane=Plane(point=_v(c), normal=_v(n), offset_m=relief_m, name="target"))


def opening_directions(g: TargetGeometry, standoff_m: float,
                       candidates: list[tuple[F64, float]]) -> list[Direction]:
    return [(a, tilt, g.c + a * standoff_m + g.dh, g.c + a * g.d_entry + g.dh)
            for a, tilt in candidates]


@dataclass(frozen=True)
class ConeChoice:
    """The direction the cone search takes, and what it saw on the way."""

    a: F64
    tilt: float
    H: F64
    E: F64
    check: LegCheck                             # the approach E → H, with r_app
    tried: list[tuple[float, float, float]]     # (tilt, heading, clearance) of each one checked
    outside: int                                # not tried: E or H too close to the grid's edge
    clear: bool                                 # False: none that fits is clear, `first` shown
    direct_fits: bool                           # the direct approach's E and H fit the grid
    note: str                                   # how it was chosen, for the approach findings


def _corridor_ok(check) -> bool:
    return check.blocked_state is None and not check.offenders


def choose_direction(grid: OccupancyGrid, directions: list[Direction], *, r_app: float,
                     r_eh: float, plane: Plane, cone_deg: float, cone_size: int) -> ConeChoice:
    """The clearest corridor among the directions whose E and H fit the grid (refused when none
    does); with none clear, the first that fits: the direct approach, or (when its E or H fall
    short) the least tilted one that fits. `cone_size`: directions of the cone, incl. the
    direct one."""
    fit = directions_inside_grid(grid, directions, r_eh, cone_deg)
    outside = len(directions) - len(fit)
    direct_fits = fit[0][1] == 0.0
    tried: list[tuple[float, float, float]] = []
    best = None
    first = None
    for a, tilt, H, E in fit:
        check = check_leg(grid, [E, H], r_app, plane)
        tried.append((tilt, heading_deg(a[:2]) if np.linalg.norm(a[:2]) > 1e-9 else 0.0,
                      check.min_clearance_m))
        entry = (a, tilt, H, E, check)
        if first is None:
            first = entry
        if _corridor_ok(check):
            if best is None or check.min_clearance_m > best[4].min_clearance_m + 1e-6:
                best = entry
            if tilt == 0.0:
                break                                    # the direct approach is clear: take it
    assert first is not None
    a, tilt, H, E, check = best if best is not None else first
    cone = f"{cone_deg:g}° cone"
    if best is None and cone_size > 1 and not outside:
        note = f"; none of the {len(tried)} directions in the {cone} is clear"
    elif best is None and cone_size > 1:
        shown = "the direct approach" if tilt == 0.0 else f"the least tilted ({tilt:.1f}°)"
        note = (f"; none of the {len(tried)} directions in the {cone} that fit the grid is "
                f"clear ({outside} more come closer to its edge than r + 2 voxels and were "
                f"not tried), {shown} is shown")
    elif tilt == 0.0:
        note = ""
    elif direct_fits:
        note = f"; direct approach blocked, tilted {tilt:.1f}° in the cone"
    else:
        note = (f"; the direct approach comes closer to the grid's edge than r + 2 voxels, "
                f"tilted {tilt:.1f}° in the cone")
    return ConeChoice(a=a, tilt=tilt, H=H, E=E, check=check, tried=tried, outside=outside,
                      clear=best is not None, direct_fits=direct_fits, note=note)


def opening_approach(target: MissionTarget, ap: AccessPoint, grid: OccupancyGrid,
                     profile: Optional[PlanningProfile] = None) -> ConeChoice:
    """The direction `plan_mission` takes to an opening, found without a home.

    Nothing in that choice depends on home: E and H hang off the opening, and the corridor
    E → H that ranks the directions is checked with r_app against the target surface. So the
    takeoff search computes it once and checks every candidate home's transit to its E.
    Refuses (outside_grid) exactly where plan_mission would from any home: the target, or E
    and H of every direction of the cone, too close to the grid's edge.
    """
    prof = profile or PlanningProfile()
    spec = target.handover
    g = target_geometry(target, ap, prof.surface_relief_m + 0.5 * SQRT3 * grid.res_m)
    if g.kind != "opening":
        raise ValueError(f"access point {ap.id} is a pad: its E depends on the transit height")
    _, r_app, r_tr = corridor_radii(target, prof)
    require_inside_grid(grid, [("target", g.c, r_app)])
    candidates = cone_directions(g.n, 0.5 * spec.approach_cone_deg, prof.cone_tilt_steps,
                                 prof.cone_azimuths)
    return choose_direction(grid, opening_directions(g, spec.standoff_m, candidates),
                            r_app=r_app, r_eh=max(r_tr, r_app), plane=g.plane,
                            cone_deg=spec.approach_cone_deg, cone_size=len(candidates))


# ── home ─────────────────────────────────────────────────────────────────
def check_home(grid: OccupancyGrid, x: float, y: float, z: Optional[float],
               relief_m: float) -> tuple[float, list[Finding]]:
    """(home height, findings) for a home at (x, y): the measured surface under it, and z if
    given. Relative altitudes are measured from home, so its height must be that surface — a
    home floating 3 m above the ground shifts every altitude of the mission by 3 m."""
    findings: list[Finding] = []
    s = grid.spec
    sz = surface_z(grid, x, y)
    i, j, _ = s.index_of([[x, y, s.z0]])
    inside = 0 <= i[0] < s.rows and 0 <= j[0] < s.cols
    source = int(grid.column_source[i[0], j[0]]) if inside else 0
    measured = inside and source in (ColumnSource.POINTS, ColumnSource.CELL, ColumnSource.CARVED)
    if sz is None or not measured:
        findings.append(_finding(Code.home_unknown, f"the surface under home ({x:.1f}, {y:.1f}) is "
                                                    f"not measured, so its height is not known",
                                 at=(x, y, 0.0)))
    if z is None:
        z = sz if sz is not None else 0.0
    elif sz is not None and abs(z - sz) > max(1.0, s.res_m):
        findings.append(_finding(
            Code.home_not_on_surface,
            f"home is at z={z:.2f} but the surface under it is at {sz:.2f}: relative "
            f"altitudes would all be off by {z - sz:+.2f} m", at=(x, y, z)))
    probe = np.array([[x, y, z + relief_m + s.res_m]])
    if int(grid.state_at(probe)[0]) != FREE:
        findings.append(_finding(Code.home_blocked, "the space just above home is not free",
                                 at=probe[0]))
    return z, findings


def resolve_home(grid: OccupancyGrid, frame: EnuFrame, home_enu: Optional[Vec3],
                 relief_m: float, source: Optional[str] = None) -> tuple[HomePoint, list[Finding]]:
    """Where the vehicle arms (check_home), with its WGS84 position and how it was chosen."""
    if home_enu is None:
        z, findings = check_home(grid, 0.0, 0.0, None, relief_m)
        x, y = 0.0, 0.0
        how = "site origin (east 0, north 0), on the measured surface"
    else:
        x, y = float(home_enu[0]), float(home_enu[1])
        z, findings = check_home(grid, x, y, float(home_enu[2]), relief_m)
        how = source or "given by the operator"
    lat, lon, h = geodetic(frame, (x, y, z))
    return HomePoint(enu=(x, y, z), lat=float(lat[0]), lon=float(lon[0]), h_ellipsoid=float(h[0]),
                     source=how), findings


# ── the plan ─────────────────────────────────────────────────────────────
def transit_altitude(grid: OccupancyGrid, xy_from, xy_to, radius_m: float,
                     extra_m: float = 0.0) -> tuple[float, Optional[Vec3]]:
    """Lowest height at which the straight transit between two points keeps `radius_m` from
    every column within `radius_m` of it (and within radius + extra of the destination)."""
    r = grid.res_m
    a = np.asarray(xy_from, float)[:2]
    b = np.asarray(xy_to, float)[:2]
    pts = sample_path(np.array([[a[0], a[1], 0.0], [b[0], b[1], 0.0]]), 0.5 * r)[:, :2]
    top, at = column_ceiling(grid, pts, radius_m)
    if extra_m > 0:
        top2, at2 = column_ceiling(grid, b[None], radius_m + extra_m)
        if top2 > top:
            top, at = top2, at2
    if not math.isfinite(top):
        return math.inf, at
    # + a voxel: the leg check wants radius + step/2, and the grid's bound is conservative.
    return math.ceil((top + radius_m + r) * 2.0) / 2.0, at


def plan_mission(*, target: MissionTarget, ap: AccessPoint, grid: OccupancyGrid,
                 occupancy: OccupancyRef, origin: GeoPoint, by: str,
                 profile: Optional[PlanningProfile] = None, home_enu: Optional[Vec3] = None,
                 altitude_frame: AltitudeFrame = AltitudeFrame.relative_home,
                 home_source: Optional[str] = None) -> MissionPlan:
    """Compute a mission for a confirmed target. Raises MissionRefused when it may not be planned.

    `home_source` says how a given `home_enu` was chosen (default: by the operator)."""
    refused = planning_findings(target, ap)
    if refused:
        raise MissionRefused(refused)
    require_relative(altitude_frame)
    prof = profile or PlanningProfile()
    spec = target.handover
    res = grid.res_m
    relief = prof.surface_relief_m + 0.5 * SQRT3 * res
    findings: list[Finding] = []

    sigma, r_app, r_tr = corridor_radii(target, prof)

    frame = EnuFrame(origin)
    home, f_home = resolve_home(grid, frame, home_enu, relief, home_source)
    findings += f_home
    hxyz = np.asarray(home.enu, float)

    # ── geometry ─────────────────────────────────────────────────────────
    half = 0.5 * spec.approach_cone_deg
    geom = target_geometry(target, ap, relief)
    kind, c, n, target_plane = geom.kind, geom.c, geom.n, geom.plane
    require_inside_grid(grid, [("home", hxyz, r_tr), ("target", c, r_app)])
    if spec.standoff_m < prof.r_drone_m + spec.min_clearance_m:
        findings.append(_finding(Code.standoff_too_small,
                                 f"standoff {spec.standoff_m:.2f} m leaves less than r_drone + m = "
                                 f"{prof.r_drone_m + spec.min_clearance_m:.2f} m to the target "
                                 f"surface", at=c))

    candidates = cone_directions(n, half, prof.cone_tilt_steps, prof.cone_azimuths)
    max_z = hxyz[2] + prof.max_alt_rel_m
    z_pad, pad_at = math.inf, None
    if kind == "pad":
        hover = c + UP * spec.standoff_m
        # The transit height above a pad depends on how far the cone may push E sideways.
        z_pad, pad_at = transit_altitude(grid, hxyz, c, r_tr)
        for _ in range(6):
            if not math.isfinite(z_pad):
                break
            reach = max(0.0, z_pad - hover[2]) * math.tan(math.radians(half))
            z_new, at = transit_altitude(grid, hxyz, c, r_tr, extra_m=reach)
            if z_new <= z_pad:
                break
            z_pad, pad_at = z_new, at
    # Where E goes when no transit height exists: the ceiling (the plan is infeasible anyway).
    z_entry = z_pad if math.isfinite(z_pad) else max_z

    # H and E of every direction the cone search may choose (a pad's flat ones are not tried);
    # those too close to the grid's edge are not tried either. For an opening nothing here
    # depends on home (`opening_approach`, which the takeoff search uses, is the same choice).
    if kind == "opening":
        directions = opening_directions(geom, spec.standoff_m, candidates)
    else:
        directions = [(a, tilt, hover, hover + a * max(0.0, z_entry - hover[2]) / a[2])
                      for a, tilt in candidates if a[2] > 0.2]
    r_eh = max(r_tr, r_app)
    choice = choose_direction(grid, directions, r_app=r_app, r_eh=r_eh, plane=target_plane,
                              cone_deg=spec.approach_cone_deg, cone_size=len(candidates))
    a, tilt, H, E, app_check = choice.a, choice.tilt, choice.H, choice.E, choice.check
    tried, outside, cone_note = choice.tried, choice.outside, choice.note

    # ── transit height ───────────────────────────────────────────────────
    if kind == "opening":
        z_t, z_at = transit_altitude(grid, hxyz, E, r_tr)
        if math.isfinite(z_t):
            z_t = max(z_t, float(E[2]))
    else:
        z_t, z_at = z_pad, pad_at
    if not math.isfinite(z_t):
        findings.append(_finding(Code.unknown_space, "the transit crosses a column that is blocked "
                                                     "all the way up (unknown, outside the grid or "
                                                     "above its ceiling)", leg="transit", at=z_at))
        z_t = max_z
    elif z_t > max_z:
        findings.append(_finding(Code.transit_too_high,
                                 f"clearing everything on the way needs {z_t - hxyz[2]:.1f} m above "
                                 f"home; the ceiling is {prof.max_alt_rel_m:g} m", leg="transit",
                                 at=z_at))
    if kind == "pad":
        E = H + a * max(0.0, z_t - H[2]) / a[2]
        app_check = check_leg(grid, [E, H], r_app, target_plane)

    # Where the chosen approach put E and H. Only directions that fit were tried, so this holds
    # unless the pad's E, placed again at the transit height above, moved.
    require_inside_grid(grid, _entry_hover(H, E, r_eh))
    home_up = np.array([hxyz[0], hxyz[1], z_t])
    e_up = np.array([E[0], E[1], z_t])
    legs: list[Leg] = []

    def add(name: str, pts, radius: float, sig: float, plane: Optional[Plane] = None,
            risk_only: bool = False, check=None, what: str = "") -> None:
        pts = [np.asarray(p, float) for p in pts]
        chk = check if check is not None else check_leg(grid, pts, radius, plane)
        legs.append(Leg(name=name, points=[_v(p) for p in pts], radius_m=radius, sigma_m=sig,
                        excluded_surface=plane.name if plane else None,
                        min_clearance_m=chk.min_clearance_m, blocked_state=chk.blocked_state,
                        assumed_free=chk.assumed_free_at is not None, risk_only=risk_only))
        findings.extend(findings_of(name, chk, radius, risk_only=risk_only, what=what))

    home_plane = Plane(point=_v(hxyz), normal=_v(UP), offset_m=relief, name="home")
    start = hxyz + UP * (relief + res)
    if z_t > start[2]:
        add("takeoff", [start, home_up], r_tr, sigma.transit_m, home_plane)
    add("transit", [home_up, e_up], r_tr, sigma.transit_m)
    if kind == "opening" and z_t - E[2] > 1e-6:
        add("descent", [e_up, E], r_tr, sigma.transit_m)
    add("approach", [E, H], r_app, sigma.approach_m, target_plane, check=app_check,
        what=f"approach{cone_note}: ")
    terminal_end = None
    if kind == "pad":
        terminal_end = c + UP * (relief + res)
        add("terminal", [H, terminal_end], r_app, sigma.approach_m, target_plane)
    # Contingencies: plain RTL climbs straight up and flies straight home; it does not retrace
    # the checked path (SmartRTL does, which is why the .parm prefers it). Findings are risks.
    h_up = np.array([H[0], H[1], z_t])
    if kind == "opening" or tilt > 0:
        add("rtl_climb", [H, h_up], r_tr, sigma.transit_m, target_plane, risk_only=True,
            what="RTL from the hover point climbs straight up: ")
        add("rtl_return", [h_up, home_up], r_tr, sigma.transit_m, target_plane, risk_only=True,
            what="RTL from the hover point flies straight home: ")

    # Hazards the twin cannot see (ACCESS_POINTS §5.8).
    if ap.attributes.get("hazard") == "power_line":
        findings.append(_finding(Code.wire_risk, "a power line is recorded at the target", at=c))
    elif kind == "opening" or (ap.pad is not None and ap.pad.building_id is None):
        low_pts = sample_path(np.array([E, H] if kind == "opening" else [E, H, terminal_end]),
                              2.0 * res)
        for p in low_pts:
            sz = surface_z(grid, float(p[0]), float(p[1]))
            if sz is not None and p[2] - sz < prof.wire_height_m:
                findings.append(_finding(
                    Code.wire_risk, f"the approach runs {p[2] - sz:.1f} m above the surface below "
                                    f"it; wires are not reconstructed (< {prof.wire_height_m:g} m)",
                    severity=Severity.risk, leg="approach", at=p))
                break

    # ── mission items ────────────────────────────────────────────────────
    descends = bool(np.linalg.norm(E - e_up) > 1e-6)
    items: list[MissionItem] = [
        MissionItem(command=NAV_TAKEOFF, role="takeoff", enu=_v(home_up)),
        MissionItem(command=DO_CHANGE_SPEED, role="transit_speed", param1=1.0,
                    param2=prof.transit_speed_mps, param3=-1.0),
        # The waypoint at E holds (param1), so the approach leg is planned at approach speed.
        MissionItem(command=NAV_WAYPOINT, role="over_entry", enu=_v(e_up),
                    param1=0.0 if descends else prof.entry_hold_s),
    ]
    if descends:
        items.append(MissionItem(command=NAV_WAYPOINT, role="entry", enu=_v(E),
                                 param1=prof.entry_hold_s))
    items.append(MissionItem(command=DO_CHANGE_SPEED, role="approach_speed", param1=1.0,
                             param2=prof.approach_speed_mps, param3=-1.0))
    heading = None
    if kind == "opening":
        heading = heading_deg(-a[:2]) if np.linalg.norm(a[:2]) > 1e-9 else None
        if heading is not None:
            # Absolute heading, shortest turn. Copter-4.7.2 starts it right after the leg to H
            # (SITL, SIM-1: commanded 90° on a northbound approach, faced 2 s before arriving);
            # the waypoint after H gives the yaw back to WP_YAW_BEHAVIOR.
            items.append(MissionItem(command=CONDITION_YAW, role="face_target", param1=heading,
                                     param2=0.0, param3=0.0, param4=0.0))
    items.append(MissionItem(command=NAV_WAYPOINT, role="hover", enu=_v(H), param1=prof.hold_s))
    if descends:
        items.append(MissionItem(command=NAV_WAYPOINT, role="exit", enu=_v(E)))
    items.append(MissionItem(command=DO_CHANGE_SPEED, role="transit_speed", param1=1.0,
                             param2=prof.transit_speed_mps, param3=-1.0))
    items.append(MissionItem(command=NAV_WAYPOINT, role="over_exit", enu=_v(e_up)))
    items.append(MissionItem(command=NAV_WAYPOINT, role="over_home", enu=_v(home_up)))
    items.append(MissionItem(command=NAV_LAND, role="land", enu=_v(hxyz)))

    # ── geofence and rally ───────────────────────────────────────────────
    vertices = [(hxyz, r_tr), (home_up, r_tr), (e_up, r_tr), (E, max(r_tr, r_app)), (H, r_app),
                (h_up, r_tr)]
    if terminal_end is not None:
        vertices.append((terminal_end, r_app))
    centres = np.array([v[0][:2] for v in vertices])
    radii = np.array([v[1] + prof.fence_margin_m for v in vertices])
    poly = fence_polygon(centres, radii)
    reach_xy = float(np.max(np.linalg.norm(poly - hxyz[:2], axis=1)))
    fence = Fence(polygon_enu=[(float(p[0]), float(p[1])) for p in poly],
                  alt_max_rel_m=math.ceil(z_t - hxyz[2] + prof.fence_alt_margin_m),
                  radius_m=max(30.0, math.ceil(reach_xy + 1.0)), margin_m=prof.fence_margin_m)

    approach = Approach(kind=kind, center=_v(c), normal=_v(n), direction=_v(a), tilt_deg=tilt,
                        hover=_v(H), entry=_v(E), heading_deg=heading, candidates=len(tried),
                        candidate_clearance_m=tried, candidates_outside_grid=outside)
    return MissionPlan(
        id=new_ulid(MISSION_PREFIX), site_key=target.site_key, created_by=by, target=target,
        access_point=ap, occupancy=occupancy, profile=prof, sigma=sigma,
        altitude_frame=altitude_frame, home=home, transit_z=float(z_t), approach=approach,
        legs=legs, items=items, fence=fence, rally_enu=[_v(home_up)],
        verdict=verdict_of(findings), findings=findings)
