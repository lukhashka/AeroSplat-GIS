"""Approach geometry and feasibility of one mission (ACCESS_POINTS §7.2, SIMULATION §3.1).

From a confirmed target and the occupancy grid of the twin:

    opening:  H = c + a·standoff + up·Δh,  E = c + a·D_entry + up·Δh,  heading = −a
    pad:      H = c + up·standoff,          E = H + a·L  at the transit height
    corridor: r = r_drone + m + k·σ_total,  σ_total² = σ_AP² + σ_georef² + σ_GNSS²

`a` is the target's normal `n` unless that corridor is blocked: then directions inside the
class's approach cone (half its apex angle each side) are tried and the clearest is taken.
Every number of the target's class — standoff, m, cone, D_entry, Δh, the σ gate — comes from
the taxonomy through the target's `HandoverSpec`; the vehicle's numbers from the profile.

The path is home → straight up to the transit height → over E → down to E → H (hold) and
back the same way, then land at home. Every leg is checked against the grid with its own
radius: σ_AP enters only the legs near the target, transit uses σ_georef ⊕ σ_GNSS. The
result is `feasible`, `feasible_with_risks` or `infeasible`, with named reasons.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from aerosplat.authored.models import AccessPoint, new_ulid
from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.mission.corridor import (SQRT3, Plane, check_leg, column_ceiling, findings_of,
                                        sample_path, surface_z)
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


# ── home ─────────────────────────────────────────────────────────────────
def resolve_home(grid: OccupancyGrid, frame: EnuFrame, home_enu: Optional[Vec3],
                 relief_m: float) -> tuple[HomePoint, list[Finding]]:
    """Where the vehicle arms. Relative altitudes are measured from here, so its height must be
    the measured surface under it — a home floating 3 m above the ground shifts every altitude
    of the mission by 3 m."""
    findings: list[Finding] = []
    x, y = (0.0, 0.0) if home_enu is None else (float(home_enu[0]), float(home_enu[1]))
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
    if home_enu is None:
        z = sz if sz is not None else 0.0
        how = "site origin (east 0, north 0), on the measured surface"
    else:
        z = float(home_enu[2])
        how = "given by the operator"
        if sz is not None and abs(z - sz) > max(1.0, s.res_m):
            findings.append(_finding(
                Code.home_not_on_surface,
                f"home is at z={z:.2f} but the surface under it is at {sz:.2f}: relative "
                f"altitudes would all be off by {z - sz:+.2f} m", at=(x, y, z)))
    probe = np.array([[x, y, z + relief_m + s.res_m]])
    if int(grid.state_at(probe)[0]) != FREE:
        findings.append(_finding(Code.home_blocked, "the space just above home is not free",
                                 at=probe[0]))
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


def _corridor_ok(check) -> bool:
    return check.blocked_state is None and not check.offenders


def plan_mission(*, target: MissionTarget, ap: AccessPoint, grid: OccupancyGrid,
                 occupancy: OccupancyRef, origin: GeoPoint, by: str,
                 profile: Optional[PlanningProfile] = None, home_enu: Optional[Vec3] = None,
                 altitude_frame: AltitudeFrame = AltitudeFrame.relative_home) -> MissionPlan:
    """Compute a mission for a confirmed target. Raises MissionRefused when it may not be planned."""
    refused = planning_findings(target, ap)
    if refused:
        raise MissionRefused(refused)
    require_relative(altitude_frame)
    prof = profile or PlanningProfile()
    spec = target.handover
    res = grid.res_m
    relief = prof.surface_relief_m + 0.5 * SQRT3 * res
    findings: list[Finding] = []

    # σ and radii (§7.2)
    s_ap, s_geo, s_gnss = target.sigma.ap.value_m, target.sigma.georef.value_m, prof.sigma_gnss_m
    sigma = SigmaUsed(ap_m=s_ap, georef_m=s_geo, gnss_m=s_gnss, target_m=math.hypot(s_ap, s_geo),
                      approach_m=math.sqrt(s_ap ** 2 + s_geo ** 2 + s_gnss ** 2),
                      transit_m=math.hypot(s_geo, s_gnss))
    r_app = prof.r_drone_m + spec.min_clearance_m + prof.k_sigma * sigma.approach_m
    r_tr = prof.r_drone_m + prof.transit_margin_m + prof.k_sigma * sigma.transit_m

    frame = EnuFrame(origin)
    home, f_home = resolve_home(grid, frame, home_enu, relief)
    findings += f_home
    hxyz = np.asarray(home.enu, float)

    # ── geometry ─────────────────────────────────────────────────────────
    half = 0.5 * spec.approach_cone_deg
    if ap.geometry is not None:                                   # an opening on a facade
        kind = "opening"
        c = np.asarray(ap.geometry.center, float)
        n = _unit(ap.geometry.normal)
        dh = UP * spec.hover_dh_m
        assert spec.entry_distance_m is not None                 # resolve_handover guarantees it
        d_entry = spec.entry_distance_m
    else:
        assert ap.pad is not None
        kind = "pad"
        c = np.asarray(ap.pad.center, float)
        n = UP.copy()
        dh = np.zeros(3)
        d_entry = 0.0
    target_plane = Plane(point=_v(c), normal=_v(n), offset_m=relief, name="target")
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

    tried: list[tuple[float, float, float]] = []
    best = None
    first = None
    for a, tilt in candidates:
        if kind == "opening":
            H = c + a * spec.standoff_m + dh
            E = c + a * d_entry + dh
        else:
            if a[2] <= 0.2:
                continue
            H = hover
            E = H + a * max(0.0, z_entry - H[2]) / a[2]
        check = check_leg(grid, [E, H], r_app, target_plane)
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
    chosen = best if best is not None else first
    a, tilt, H, E, app_check = chosen
    if best is None and len(candidates) > 1:
        cone_note = f"; none of the {len(tried)} directions in the {spec.approach_cone_deg:g}° cone is clear"
    else:
        cone_note = "" if tilt == 0.0 else f"; direct approach blocked, tilted {tilt:.1f}° in the cone"

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
    items: list[MissionItem] = [
        MissionItem(command=NAV_TAKEOFF, role="takeoff", enu=_v(home_up)),
        MissionItem(command=DO_CHANGE_SPEED, role="transit_speed", param1=1.0,
                    param2=prof.transit_speed_mps, param3=-1.0),
        MissionItem(command=NAV_WAYPOINT, role="over_entry", enu=_v(e_up)),
    ]
    if np.linalg.norm(E - e_up) > 1e-6:
        items.append(MissionItem(command=NAV_WAYPOINT, role="entry", enu=_v(E)))
    items.append(MissionItem(command=DO_CHANGE_SPEED, role="approach_speed", param1=1.0,
                             param2=prof.approach_speed_mps, param3=-1.0))
    heading = None
    if kind == "opening":
        heading = heading_deg(-a[:2]) if np.linalg.norm(a[:2]) > 1e-9 else None
        if heading is not None:
            # Copter: executed while flying to the next waypoint; absolute, shortest turn.
            items.append(MissionItem(command=CONDITION_YAW, role="face_target", param1=heading,
                                     param2=0.0, param3=0.0, param4=0.0))
    items.append(MissionItem(command=NAV_WAYPOINT, role="hover", enu=_v(H), param1=prof.hold_s))
    if np.linalg.norm(E - e_up) > 1e-6:
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
                        candidate_clearance_m=tried)
    return MissionPlan(
        id=new_ulid(MISSION_PREFIX), site_key=target.site_key, created_by=by, target=target,
        access_point=ap, occupancy=occupancy, profile=prof, sigma=sigma,
        altitude_frame=altitude_frame, home=home, transit_z=float(z_t), approach=approach,
        legs=legs, items=items, fence=fence, rally_enu=[_v(home_up)],
        verdict=verdict_of(findings), findings=findings)
