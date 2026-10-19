"""Oracles of the L2 runner (SIMULATION_AND_DELIVERY §4.6), judged on the TRUE trajectory.

Every oracle takes a `Recording` (truth from SIM_STATE in Site ENU, events on the simulation
clock) and the context it is judged against, and returns a `Verdict`: pass / fail / n/a, a
sentence, and numbers. None of them reads the EKF estimate.

    clearance        distance from the true position to blocked space of the twin — occupied,
                     UNKNOWN (= occupied) or outside the grid — at least r_drone + m_min. The
                     surfaces the mission touches on purpose are left out where it touches them,
                     like the planner does: the ground under home and under the touchdown point,
                     the target's own surface along the approach (its distance is checked
                     separately, against the class's minimum clearance).
    fence            inside the inclusion polygon that was uploaded, under FENCE_ALT_MAX, inside
                     FENCE_RADIUS.
    hover            came within tolerance of H and held it; approached no faster than planned.
    heading          faced the target at H, already when arriving (CONDITION_YAW placement).
    altitude_frame   home where the plan put it; every reached waypoint at its planned height
                     above home (an AMSL/relative mix-up shows as an error of the home altitude).
    failsafe         the mode the .parm prescribes for the trigger, within its delay + T_react,
                     not before the trigger, and no return to AUTO; no failsafe at all otherwise.
    completion       complete: every waypoint reached, landed and disarmed at home. abort: the
                     mission did not complete, the vehicle came down and disarmed (at home after
                     an RTL). A run still airborne at its timeout fails.
    wall_descent     never descending with a wall (occupied / unknown) closer than X
                     horizontally (H-011); X = r_drone + the transit margin unless given.
    navigation       GNSS-free runs (SIM-2n) only: no GNSS fix where there is none, the EKF
                     origin is the Site origin, the EKF estimate stays on the truth while VPS has
                     a fix (on optical flow: reported), a GNSS spoof is flagged in time. This one
                     reads the estimate — to compare it with the truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from aerosplat.mission.corridor import Plane, masked_clearance, surface_z
from aerosplat.mission.export import POSITIONAL, WireItem, distance_to_boundary, point_in_polygon
from aerosplat.mission.models import GnssUse, MissionPlan
from aerosplat.sim.faults import Response
from aerosplat.sim.record import Recording
from aerosplat.sim.scenario import Expect, Invariant, Outcome, Tolerances
from aerosplat.surface.models import OCCUPIED, UNKNOWN, state_name
from aerosplat.surface.occupancy import OccupancyGrid

F64 = NDArray[np.float64]
SQRT3 = math.sqrt(3.0)
NAV_TAKEOFF, NAV_LAND, CONDITION_YAW = 22, 21, 115
GROUND_HEIGHT_M = 0.5        # truth this close above the surface: on the ground
FLIGHT_MODES_OF_A_MISSION = {"AUTO"}
CHUNK = 64                   # samples per exact (masked) clearance window
EXACT_REACH_M = 4.0          # near touched surfaces, clearance is exact up to this (else "at least")

PASS, FAIL, NA = "pass", "fail", "n/a"


@dataclass
class Verdict:
    invariant: str
    status: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PASS


@dataclass
class NavContext:
    """What the companion saw and did in a GNSS-free run (SIM-2n), for `navigation`."""

    gnss: GnssUse
    flow: bool
    fix: list[tuple[float, bool]]           # (capture time, VPS has a fix) on every change
    gnss_fixes: int                         # GPS_RAW_INT with a 2D fix or better
    spoof_t: Optional[float] = None         # when the companion flagged GNSS as spoofed
    spoof_injected_t: Optional[float] = None
    origin_ok: bool = False
    origin_required: bool = True
    summary: dict[str, Any] = field(default_factory=dict)

    def has_fix(self, t: F64) -> NDArray[np.bool_]:
        out = np.zeros(len(t), bool)
        for k, (t0, fix) in enumerate(self.fix):
            t1 = self.fix[k + 1][0] if k + 1 < len(self.fix) else math.inf
            if fix:
                out |= (t >= t0) & (t < t1)
        return out


@dataclass
class Context:
    """What a run is judged against."""

    plan: MissionPlan
    items: list[WireItem]                   # the uploaded mission, seq 0 = home
    planned_enu: dict[int, tuple[float, float, float]]   # positional items in Site ENU
    fence_enu: list[tuple[float, float]]    # the uploaded inclusion polygon, Site ENU
    params: dict[str, float]                # what the vehicle flew with (live + package)
    grid: OccupancyGrid
    expect: Expect
    response: Optional[Response] = None     # what the .parm makes of the expected trigger
    trigger_t: Optional[float] = None       # when that trigger was injected (sim s)
    mission_start_t: Optional[float] = None
    nav: Optional[NavContext] = None        # GNSS-free runs only

    @property
    def tol(self) -> Tolerances:
        return self.expect.tolerances

    @property
    def home(self) -> F64:
        return np.asarray(self.plan.home.enu, float)

    def seq_of(self, role: str) -> Optional[int]:
        return next((k for k, it in enumerate(self.plan.items, start=1) if it.role == role), None)

    @property
    def required_clearance_m(self) -> float:
        m = self.expect.min_clearance_m
        if m is None:
            m = min(self.plan.target.handover.min_clearance_m, self.plan.profile.transit_margin_m)
        return self.plan.profile.r_drone_m + m


# ── helpers ──────────────────────────────────────────────────────────────
def airborne_slice(rec: Recording) -> slice:
    """Samples from the first armed one to the last armed one."""
    armed = np.flatnonzero(np.asarray(rec.armed, bool))
    if len(armed) == 0:
        return slice(0, 0)
    return slice(int(armed[0]), int(armed[-1]) + 1)


def _dist_to_segments(p: F64, pts: F64) -> F64:
    """Distance from each p to the polyline pts."""
    best = np.full(len(p), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(p - (a + t[:, None] * ab), axis=1))
    return best


def _wrap180(a: F64) -> F64:
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is true."""
    out, start = [], None
    for k, m in enumerate(mask):
        if m and start is None:
            start = k
        elif not m and start is not None:
            out.append((start, k))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


# ── clearance ────────────────────────────────────────────────────────────
@dataclass
class ClearanceSeries:
    t: F64
    clearance: F64          # to blocked space, with the touched surfaces left out
    target_distance: F64    # signed distance to the target plane, NaN outside the approach zone
    required: float
    state: list[str]        # state of the voxel the true position is in
    cap: float = math.inf   # where a touched surface is left out, values stop at this ("at least")


def clearance_series(rec: Recording, ctx: Context) -> ClearanceSeries:
    sl = airborne_slice(rec)
    p = rec.positions[sl]
    t = rec.times[sl]
    grid = ctx.grid
    res = grid.res_m
    req = ctx.required_clearance_m
    prof = ctx.plan.profile
    relief = prof.surface_relief_m + 0.5 * SQRT3 * res
    if len(p) == 0:
        return ClearanceSeries(t, np.zeros(0), np.zeros(0), req, [])

    # Ground planes: under home, and under where the vehicle came down. Only over those spots
    # (within `zone_r`); there the distance is found by brute force, exact up to `reach`.
    zone_r = req + res
    reach = max(zone_r, EXACT_REACH_M)
    planes: list[tuple[NDArray[np.bool_], Plane]] = []
    spots = [ctx.home]
    touchdown = p[-1]
    if np.linalg.norm(touchdown[:2] - ctx.home[:2]) > zone_r:
        spots.append(touchdown)
    for s in spots:
        sz = surface_z(grid, float(s[0]), float(s[1]))
        z = float(s[2]) if sz is None else sz
        near = np.linalg.norm(p[:, :2] - s[None, :2], axis=1) <= zone_r
        planes.append((near, Plane(point=(float(s[0]), float(s[1]), z), normal=(0.0, 0.0, 1.0),
                                   offset_m=relief, name="ground")))
    # The target's surface, along the legs that end on it (approach, pad terminal descent).
    ap = ctx.plan.approach
    tgt_plane = Plane(point=ap.center, normal=ap.normal, offset_m=relief, name="target")
    zone = np.zeros(len(p), bool)
    for leg in ctx.plan.legs:
        if leg.excluded_surface == "target" and not leg.risk_only:
            zone |= _dist_to_segments(p, np.asarray(leg.points, float)) <= leg.radius_m + zone_r
    if zone.any():
        planes.append((zone, tgt_plane))

    c = grid.clearance(p)
    excluded = np.zeros(len(p), bool)
    for near, _ in planes:
        excluded |= near
    idx = np.flatnonzero(excluded)
    for a in range(0, len(idx), CHUNK):
        chunk = idx[a:a + CHUNK]
        q = p[chunk]
        best = np.full(len(chunk), np.inf)
        # Each sample leaves out only the planes it is over; group by that combination.
        masks = np.array([[near[k] for near, _ in planes] for k in chunk])
        for combo in {tuple(m) for m in masks}:
            sel = np.all(masks == np.array(combo), axis=1)
            excl = [pl for (_, pl), on in zip(planes, combo) if on]
            cl = _masked(grid, q[sel], reach, excl)
            best[sel] = cl
        c[chunk] = np.minimum(best, np.maximum(c[chunk], reach))
    # Inside blocked space — unless the voxel is the ground the vehicle stands on (it sits in
    # the 2.5-D relief of the surface under home or under its touchdown point). The target's
    # surface is never touched: being inside it is a penetration.
    codes = grid.state_at(p)
    spec = grid.spec
    centres = spec.centre_of(*spec.index_of(p))
    touched = np.zeros(len(p), bool)
    for near, pl in planes:
        if pl.name == "ground":
            d = (centres - np.asarray(pl.point)) @ np.asarray(pl.normal)
            touched |= near & (d <= pl.offset_m) & (d >= -res)   # the relief band, not below it
    states = [state_name(int(code)) if not on else "free" for code, on in zip(codes, touched)]
    c[np.array([st != "free" for st in states])] = 0.0

    n = np.asarray(ap.normal, float)
    tdist = np.full(len(p), np.nan)
    if zone.any():
        tdist[zone] = (p[zone] - np.asarray(ap.center, float)) @ n
    return ClearanceSeries(t, c, tdist, req, states, cap=reach if excluded.any() else math.inf)


def _masked(grid: OccupancyGrid, q: F64, reach: float, planes: list[Plane]) -> F64:
    """Exact clearance (capped at `reach`) leaving out everything behind ANY of `planes`."""
    if len(planes) == 1:
        cl, _, _, _ = masked_clearance(grid, q, reach, planes[0])
        return np.minimum(cl, reach)
    # Several planes: drop voxels behind any of them (corridor.masked_clearance takes one).
    from aerosplat.mission.corridor import _blocked_in_window

    r = grid.res_m
    lo, hi = q.min(axis=0) - reach - r, q.max(axis=0) + reach + r
    centres, _ = _blocked_in_window(grid, lo, hi)
    keep = np.ones(len(centres), bool)
    for pl in planes:
        keep &= (centres - np.asarray(pl.point)) @ np.asarray(pl.normal) > pl.offset_m
    centres = centres[keep]
    if len(centres) == 0:
        return np.full(len(q), reach)
    gap = np.abs(q[:, None, :] - centres[None, :, :])
    d = np.sqrt((np.maximum(gap - 0.5 * r, 0.0) ** 2).sum(-1)).min(axis=1)
    return np.minimum(d, reach)


def check_clearance(rec: Recording, ctx: Context, series: Optional[ClearanceSeries] = None) -> Verdict:
    s = series or clearance_series(rec, ctx)
    if len(s.t) == 0:
        return Verdict(Invariant.clearance.value, NA, "never airborne")
    k = int(np.argmin(s.clearance))
    below = s.clearance < s.required
    p = rec.positions[airborne_slice(rec)]
    entered = [i for i, st in enumerate(s.state) if st != "free"]
    min_tgt = float(np.nanmin(s.target_distance)) if np.isfinite(s.target_distance).any() else None
    tgt_req = ctx.plan.profile.r_drone_m + ctx.plan.target.handover.min_clearance_m
    metrics = {"min_clearance_m": float(s.clearance[k]), "at_t": float(s.t[k]),
               "at": [round(float(v), 2) for v in p[k]], "required_m": s.required,
               "seconds_below": float(np.sum(np.diff(s.t, prepend=s.t[0])[below])),
               "samples": int(len(s.t)), "min_target_distance_m": min_tgt,
               "target_required_m": tgt_req}
    if entered:
        i = entered[0]
        metrics["entered"] = {"t": float(s.t[i]), "state": s.state[i],
                              "at": [round(float(v), 2) for v in p[i]]}
        return Verdict(Invariant.clearance.value, FAIL,
                       f"the true trajectory enters {s.state[i]} space at t={s.t[i]:.1f} s "
                       f"({p[i][0]:.1f}, {p[i][1]:.1f}, {p[i][2]:.1f})", metrics)
    if below.any():
        return Verdict(Invariant.clearance.value, FAIL,
                       f"clearance {s.clearance[k]:.2f} m < required {s.required:.2f} m at "
                       f"t={s.t[k]:.1f} s ({p[k][0]:.1f}, {p[k][1]:.1f}, {p[k][2]:.1f}), "
                       f"{metrics['seconds_below']:.1f} s below", metrics)
    if min_tgt is not None and min_tgt < tgt_req:
        return Verdict(Invariant.clearance.value, FAIL,
                       f"came {min_tgt:.2f} m from the target surface; the class needs "
                       f"r_drone + m = {tgt_req:.2f} m", metrics)
    at_least = "≥ " if s.clearance[k] >= s.cap - 1e-6 else ""
    return Verdict(Invariant.clearance.value, PASS,
                   f"min clearance {at_least}{s.clearance[k]:.2f} m ≥ {s.required:.2f} m"
                   + (f"; target surface ≥ {min_tgt:.2f} m" if min_tgt is not None else ""), metrics)


# ── fence ────────────────────────────────────────────────────────────────
def fence_excursions(rec: Recording, ctx: Context) -> tuple[F64, F64]:
    """(times, how far outside [m]) per airborne sample; 0 inside. Horizontal and vertical."""
    sl = airborne_slice(rec)
    p = rec.positions[sl]
    t = rec.times[sl]
    poly = ctx.fence_enu
    out = np.zeros(len(p))
    ftype = int(ctx.params.get("FENCE_TYPE", 7))
    alt_max = float(ctx.params.get("FENCE_ALT_MAX", math.inf))
    radius = float(ctx.params.get("FENCE_RADIUS", math.inf))
    h = ctx.home
    for k, q in enumerate(p):
        d = 0.0
        if ftype & 4 and poly and not point_in_polygon(float(q[0]), float(q[1]), poly):
            d = max(d, distance_to_boundary(float(q[0]), float(q[1]), poly))
        if ftype & 1:
            d = max(d, float(q[2] - h[2]) - alt_max)
        if ftype & 2:
            d = max(d, float(np.linalg.norm(q[:2] - h[:2])) - radius)
        out[k] = max(d, 0.0)
    return t, out


def check_fence(rec: Recording, ctx: Context) -> Verdict:
    t, out = fence_excursions(rec, ctx)
    if len(t) == 0:
        return Verdict(Invariant.fence.value, NA, "never airborne")
    k = int(np.argmax(out))
    m = {"max_outside_m": float(out[k]), "first_breach_t": None,
         "fence_alt_max_m": ctx.params.get("FENCE_ALT_MAX"), "vertices": len(ctx.fence_enu)}
    if out[k] > 0:
        first = int(np.argmax(out > 0))
        m["first_breach_t"] = float(t[first])
        return Verdict(Invariant.fence.value, FAIL,
                       f"outside the fence from t={t[first]:.1f} s, up to {out[k]:.1f} m", m)
    return Verdict(Invariant.fence.value, PASS, "inside the inclusion polygon, altitude and radius", m)


def fence_breach_time(rec: Recording, ctx: Context) -> Optional[float]:
    t, out = fence_excursions(rec, ctx)
    return float(t[int(np.argmax(out > 0))]) if (out > 0).any() else None


# ── hover and heading ────────────────────────────────────────────────────
def _hold_window(rec: Recording, ctx: Context) -> tuple[Optional[int], F64, F64, F64, list[tuple[int, int]]]:
    seq = ctx.seq_of("hover")
    H = np.asarray(ctx.plan.approach.hover, float)
    p = rec.positions
    flying_to = np.asarray(rec.seq, int) == (seq if seq is not None else -1)
    dh = np.linalg.norm(p[:, :2] - H[None, :2], axis=1)
    dz = np.abs(p[:, 2] - H[2])
    inside = flying_to & (dh <= ctx.tol.hover_m) & (dz <= ctx.tol.hover_alt_m)
    return seq, dh, dz, rec.times, _runs(inside)


def check_hover(rec: Recording, ctx: Context) -> Verdict:
    seq, dh, dz, t, runs = _hold_window(rec, ctx)
    if seq is None:
        return Verdict(Invariant.hover.value, NA, "the mission has no hover item")
    flying = np.asarray(rec.seq, int) == seq
    if not flying.any():
        return Verdict(Invariant.hover.value, FAIL, "the hover item was never flown to")
    hold = min(ctx.tol.hover_hold_s, ctx.plan.profile.hold_s)
    best = max(runs, key=lambda r: t[r[1] - 1] - t[r[0]], default=None)
    held = 0.0 if best is None else float(t[best[1] - 1] - t[best[0]])
    m = {"min_horizontal_m": float(dh[flying].min()), "min_vertical_m": float(dz[flying].min()),
         "held_s": held, "required_hold_s": hold, "tolerance_m": ctx.tol.hover_m,
         "tolerance_alt_m": ctx.tol.hover_alt_m}
    # The approach itself: no faster than planned (onboard guidance has to keep up near the
    # target). Horizontal ground speed of the truth from the start of the leg to arrival.
    vel = np.asarray(rec.vel_ned, float).reshape(-1, 3)
    upto = np.flatnonzero(flying)
    if best is not None:
        upto = upto[upto < best[0]]
        m["hold_mean_horizontal_m"] = float(dh[best[0]:best[1]].mean())
        m["hold_max_vertical_m"] = float(dz[best[0]:best[1]].max())
        m["arrived_t"] = float(t[best[0]])
    # DO_CHANGE_SPEED sets the ground (horizontal) speed; a vertical approach (pad) descends
    # at WP_SPD_DN, which is reported but not judged here.
    planned_v = ctx.plan.profile.approach_speed_mps
    v_max = float(np.hypot(vel[upto, 0], vel[upto, 1]).max()) if len(upto) else 0.0
    m["approach_max_speed_mps"], m["approach_planned_speed_mps"] = v_max, planned_v
    m["approach_max_vertical_mps"] = float(np.abs(vel[upto, 2]).max()) if len(upto) else 0.0
    if v_max > planned_v * ctx.tol.approach_speed_ratio:
        return Verdict(Invariant.hover.value, FAIL,
                       f"approached H at up to {v_max:.2f} m/s; planned {planned_v:g} m/s "
                       f"(allowed ×{ctx.tol.approach_speed_ratio:g})", m)
    if best is None or held + 1e-6 < hold:
        return Verdict(Invariant.hover.value, FAIL,
                       f"held H for {held:.1f} s within {ctx.tol.hover_m:g} m / "
                       f"{ctx.tol.hover_alt_m:g} m (need {hold:g} s); closest "
                       f"{m['min_horizontal_m']:.2f} m horizontally", m)
    return Verdict(Invariant.hover.value, PASS,
                   f"held H for {held:.1f} s, mean {m['hold_mean_horizontal_m']:.2f} m off, "
                   f"≤ {m['hold_max_vertical_m']:.2f} m vertically; approach ≤ {v_max:.2f} m/s horizontally, "
                   f"{m['approach_max_vertical_mps']:.2f} m/s vertically", m)


def commanded_heading(ctx: Context) -> Optional[tuple[int, float]]:
    """(seq, heading) of the CONDITION_YAW before the hover item, if the mission has one."""
    hover = ctx.seq_of("hover")
    if hover is None:
        return None
    for it in reversed(ctx.items[1:hover]):
        if it.command == CONDITION_YAW:
            return it.seq, float(it.params[0])
        if it.command in POSITIONAL:
            return None
    return None


def check_heading(rec: Recording, ctx: Context) -> Verdict:
    cmd = commanded_heading(ctx)
    if cmd is None:
        return Verdict(Invariant.heading.value, NA, "no CONDITION_YAW before the hover item")
    _, heading = cmd
    seq, dh, dz, t, runs = _hold_window(rec, ctx)
    if not runs:
        return Verdict(Invariant.heading.value, FAIL, "never held H, so never faced the target there")
    yaw = np.asarray(rec.yaw_deg, float)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    err = np.abs(_wrap180(yaw[a:b] - heading))
    flying = np.flatnonzero(np.asarray(rec.seq, int) == seq)
    e_all = np.abs(_wrap180(yaw[flying] - heading))
    ok_from = flying[np.argmax(e_all <= ctx.tol.heading_deg)] if (e_all <= ctx.tol.heading_deg).any() else None
    m = {"commanded_deg": heading, "max_error_at_hold_deg": float(err.max()),
         "error_on_arrival_deg": float(err[0]), "arrived_t": float(t[a]),
         "faced_from_t": None if ok_from is None else float(t[ok_from]),
         "faced_before_arrival_s": None if ok_from is None else float(t[a] - t[ok_from])}
    if err[0] > ctx.tol.heading_deg:
        return Verdict(Invariant.heading.value, FAIL,
                       f"arrived at H facing {err[0]:.0f}° away from {heading:.0f}°: the yaw "
                       f"command did not act on the approach leg", m)
    if err.max() > ctx.tol.heading_deg:
        return Verdict(Invariant.heading.value, FAIL,
                       f"heading drifted {err.max():.0f}° from {heading:.0f}° while holding H", m)
    return Verdict(Invariant.heading.value, PASS,
                   f"faced {heading:.0f}° ± {err.max():.1f}° at H, from "
                   f"{m['faced_before_arrival_s']:.1f} s before arriving", m)


# ── altitude frame ───────────────────────────────────────────────────────
def check_altitude_frame(rec: Recording, ctx: Context, preflight: Optional[dict] = None) -> Verdict:
    pre = preflight or {}
    m: dict[str, Any] = {"upload_verified": pre.get("upload_verified"),
                         "frames": pre.get("frames"), "items": []}
    if pre.get("upload_verified") is False:
        return Verdict(Invariant.altitude_frame.value, FAIL, "the vehicle does not hold the uploaded "
                                                            "mission", m)
    sl = airborne_slice(rec)
    if sl.stop == 0:
        return Verdict(Invariant.altitude_frame.value, NA, "never airborne", m)
    p = rec.positions
    t = rec.times
    seq = np.asarray(rec.seq, int)
    hz = float(ctx.home[2])
    # The vehicle armed where the plan's home is: relative altitudes start from there.
    z0 = float(p[sl.start, 2])
    m["home_offset_m"] = z0 - hz
    m["home_horizontal_m"] = float(np.linalg.norm(p[sl.start, :2] - ctx.home[:2]))
    worst = 0.0
    problems: list[str] = []
    if m["home_horizontal_m"] > ctx.tol.land_m or abs(m["home_offset_m"]) > ctx.tol.altitude_m:
        problems.append(f"armed {m['home_horizontal_m']:.1f} m / {m['home_offset_m']:+.2f} m from "
                        f"the plan's home")
    reached = {int(e.data.get("seq", -1)) for e in rec.events_of("reached")}
    for it in ctx.items[1:]:
        if it.command not in POSITIONAL or it.command == NAV_LAND or it.seq not in reached:
            continue
        want = np.asarray(ctx.planned_enu[it.seq], float)
        # The samples flying to this item and up to two seconds after it was reached.
        rt = next(e.t for e in rec.events_of("reached") if e.data.get("seq") == it.seq)
        sel = np.flatnonzero((seq == it.seq) | ((t >= rt) & (t <= rt + 2.0)))
        if len(sel) == 0:
            continue
        k = sel[int(np.argmin(np.linalg.norm(p[sel] - want, axis=1)))]
        dz = float(p[k, 2] - want[2])
        dxy = float(np.linalg.norm(p[k, :2] - want[:2]))
        m["items"].append({"seq": it.seq, "planned_rel_m": it.alt, "flown_rel_m": float(p[k, 2] - hz),
                           "dz_m": dz, "dxy_m": dxy})
        worst = max(worst, abs(dz))
        if abs(dz) > ctx.tol.altitude_m:
            problems.append(f"item {it.seq}: flew {p[k, 2] - hz:.2f} m above home, planned "
                            f"{it.alt:.2f} m ({dz:+.2f} m)")
    m["worst_dz_m"] = worst
    if not m["items"]:
        problems.append("no positional waypoint was reached, nothing to compare")
    if problems:
        return Verdict(Invariant.altitude_frame.value, FAIL, "; ".join(problems), m)
    return Verdict(Invariant.altitude_frame.value, PASS,
                   f"{len(m['items'])} waypoints flown at their planned height above home "
                   f"(worst {worst:.2f} m)", m)


# ── failsafe and completion ──────────────────────────────────────────────
def modes_after(rec: Recording, t0: float) -> list[tuple[float, str]]:
    return [(t, mode) for t, mode in rec.mode_changes() if t >= t0]


def check_failsafe(rec: Recording, ctx: Context, trigger_t: Optional[float] = None) -> Verdict:
    """`trigger_t`: when the expected abort's trigger happened (default: ctx.trigger_t)."""
    if trigger_t is None:
        trigger_t = ctx.trigger_t
    start = ctx.mission_start_t
    if start is None:
        return Verdict(Invariant.failsafe.value, FAIL, "the mission never started")
    seq_modes = modes_after(rec, start)
    m: dict[str, Any] = {"modes": [[round(t, 2), md] for t, md in seq_modes], "trigger_t": trigger_t}
    off_mission = [(t, md) for t, md in seq_modes if md not in FLIGHT_MODES_OF_A_MISSION]
    disarm = rec.first("disarmed", after=start)
    if ctx.expect.outcome is not Outcome.abort:
        if off_mission:
            t, md = off_mission[0]
            cause = _cause(rec, t)
            return Verdict(Invariant.failsafe.value, FAIL,
                           f"unexpected switch to {md} at t={t:.1f} s{cause}", m)
        return Verdict(Invariant.failsafe.value, PASS, "no failsafe: AUTO from start to landing", m)
    resp = ctx.response
    assert resp is not None
    m["expected_modes"] = list(resp.modes)
    m["delay_s"] = resp.delay_s
    if trigger_t is None:
        return Verdict(Invariant.failsafe.value, FAIL,
                       f"the trigger ({resp.trigger}) never happened", m)
    early = [(t, md) for t, md in off_mission if t < trigger_t]
    if early:
        t, md = early[0]
        return Verdict(Invariant.failsafe.value, FAIL,
                       f"{md} at t={t:.1f} s, before the trigger at {trigger_t:.1f} s{_cause(rec, t)}", m)
    after = [(t, md) for t, md in off_mission if t >= trigger_t]
    if not after:
        return Verdict(Invariant.failsafe.value, FAIL,
                       f"no failsafe after {resp.trigger} at t={trigger_t:.1f} s; expected "
                       f"{' / '.join(resp.modes)} ({', '.join(resp.by)})", m)
    t1, md1 = after[0]
    react = t1 - trigger_t
    m["first_mode"], m["react_s"] = md1, react
    # How far the vehicle went between the trigger and the reaction (e.g. blind, on GPS loss).
    tt = rec.times
    pts = rec.positions[(tt >= trigger_t) & (tt <= t1)]
    m["flown_before_reaction_m"] = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) \
        if len(pts) > 1 else 0.0
    limit = resp.delay_s + ctx.tol.react_s
    wanted = tuple(ctx.expect.modes) if ctx.expect.modes else resp.modes
    if md1 not in resp.modes or md1 not in wanted:
        return Verdict(Invariant.failsafe.value, FAIL,
                       f"switched to {md1} {react:.1f} s after {resp.trigger}; the package "
                       f"prescribes {' / '.join(resp.modes)} ({', '.join(resp.by)})", m)
    if react > limit:
        return Verdict(Invariant.failsafe.value, FAIL,
                       f"{md1} came {react:.1f} s after {resp.trigger}; allowed "
                       f"{resp.delay_s:g} s delay + {ctx.tol.react_s:g} s", m)
    back = [(t, md) for t, md in seq_modes if t > t1 and md == "AUTO"
            and (disarm is None or t < disarm.t)]
    if back:
        return Verdict(Invariant.failsafe.value, FAIL,
                       f"back in AUTO at t={back[0][0]:.1f} s after the failsafe", m)
    fallback = "" if md1 == resp.modes[0] else f" (fallback; preferred {resp.modes[0]})"
    return Verdict(Invariant.failsafe.value, PASS,
                   f"{resp.trigger} at t={trigger_t:.1f} s -> {md1}{fallback} after {react:.1f} s, "
                   f"{m['flown_before_reaction_m']:.1f} m flown meanwhile (≤ {limit:g} s; "
                   f"{', '.join(resp.by)})", m)


def _cause(rec: Recording, t: float) -> str:
    near = [e.detail for e in rec.events_of("statustext") if t - 3.0 <= e.t <= t + 0.5]
    return f" ({'; '.join(near[-3:])})" if near else ""


def check_completion(rec: Recording, ctx: Context) -> Verdict:
    m: dict[str, Any] = {}
    end = rec.first("end")
    if end is not None:
        m["end"] = end.detail
    if not len(rec):
        return Verdict(Invariant.completion.value, FAIL, "no truth recorded", m)
    p_end = rec.positions[-1]
    sz = surface_z(ctx.grid, float(p_end[0]), float(p_end[1]))
    ground = float(ctx.home[2]) if sz is None else sz
    m["final"] = [round(float(v), 2) for v in p_end]
    m["final_height_m"] = float(p_end[2] - ground)
    m["final_from_home_m"] = float(np.linalg.norm(p_end[:2] - ctx.home[:2]))
    disarmed = bool(rec.armed) and not rec.armed[-1] and any(rec.armed)
    m["disarmed"] = disarmed
    land_seq = next((it.seq for it in ctx.items if it.command == NAV_LAND), None)
    reached = [int(e.data.get("seq", -1)) for e in rec.events_of("reached")]
    nav = [it.seq for it in ctx.items[1:] if it.command in POSITIONAL]
    m["reached"] = reached
    off_mission = [md for _, md in modes_after(rec, ctx.mission_start_t or 0.0)
                   if md not in FLIGHT_MODES_OF_A_MISSION]
    land_current = any(e.data.get("seq") == land_seq for e in rec.events_of("current"))
    # Completed: the LAND item reported reached, or flown in AUTO down to the disarm.
    completed = land_seq is not None and (land_seq in reached
                                          or (land_current and disarmed and not off_mission))
    m["mission_completed"] = completed
    if end is not None and end.detail == "timeout":
        return Verdict(Invariant.completion.value, FAIL,
                       f"still {'armed' if rec.armed[-1] else 'running'} at the run's timeout, "
                       f"{m['final_height_m']:.1f} m above the surface", m)
    # Down = disarmed and at rest. Where the simulator's ground is not the twin's (L2: flat at
    # home's altitude), the touchdown height above the twin's surface is reported, not judged:
    # the clearance oracle already fails a touchdown inside something.
    vz = abs(float(rec.vel_ned[-1][2]))
    at_rest = vz < 0.3
    if abs(m["final_height_m"]) > GROUND_HEIGHT_M:
        m["ground_mismatch_m"] = m["final_height_m"]
    if not disarmed or not at_rest:
        return Verdict(Invariant.completion.value, FAIL,
                       f"did not come down: {'disarmed' if disarmed else 'armed'}, descending at "
                       f"{vz:.1f} m/s, {m['final_height_m']:.1f} m above the surface", m)
    if ctx.expect.outcome is Outcome.abort:
        if completed:
            return Verdict(Invariant.completion.value, FAIL,
                           "the mission completed although it should have been aborted", m)
        if off_mission and off_mission[0] in ("RTL", "SMART_RTL", "AUTO_RTL") \
                and m["final_from_home_m"] > ctx.tol.land_m:
            return Verdict(Invariant.completion.value, FAIL,
                           f"{off_mission[0]} but landed {m['final_from_home_m']:.1f} m from home", m)
        return Verdict(Invariant.completion.value, PASS,
                       f"aborted and came down {m['final_from_home_m']:.1f} m from home, disarmed", m)
    missing = [s for s in nav if s not in reached and s != land_seq]
    if missing:
        return Verdict(Invariant.completion.value, FAIL, f"waypoints {missing} never reached", m)
    if not completed:
        return Verdict(Invariant.completion.value, FAIL, "the final LAND item did not complete", m)
    if m["final_from_home_m"] > ctx.tol.land_m:
        return Verdict(Invariant.completion.value, FAIL,
                       f"landed {m['final_from_home_m']:.1f} m from home (> {ctx.tol.land_m:g} m)", m)
    return Verdict(Invariant.completion.value, PASS,
                   f"all {len(nav)} waypoints, landed {m['final_from_home_m']:.2f} m from home, disarmed", m)


# ── walls while descending (H-011) ───────────────────────────────────────
WALL_RAYS = 16               # horizontal directions searched for a wall around a sample
WALL_MIN_HEIGHT_M = 1.0      # below this above the surface, horizontal rays meet ground relief


def wall_distance(grid: OccupancyGrid, p: F64, reach_m: float) -> F64:
    """Horizontal distance from each point to the nearest occupied or unknown voxel within
    `reach_m` (inf: none). The grid's edge is not a wall."""
    p = np.asarray(p, float).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros(0)
    a = np.linspace(0.0, 2.0 * math.pi, WALL_RAYS, endpoint=False)
    dirs = np.column_stack([np.cos(a), np.sin(a), np.zeros(WALL_RAYS)])
    hits = grid.raycast(np.repeat(p, WALL_RAYS, axis=0), np.tile(dirs, (len(p), 1)), reach_m)
    d = np.where(np.isin(hits.state, (OCCUPIED, UNKNOWN)), hits.distance, np.inf)
    return d.reshape(len(p), WALL_RAYS).min(axis=1)


def check_wall_descent(rec: Recording, ctx: Context) -> Verdict:
    """Never descend with a wall closer than X horizontally (H-011: a landing that drifts into
    the facade). X: the scenario's, else r_drone + the profile's transit margin."""
    prof = ctx.plan.profile
    x_req = ctx.tol.wall_descent_m or prof.r_drone_m + prof.transit_margin_m
    sl = airborne_slice(rec)
    p, t = rec.positions[sl], rec.times[sl]
    m: dict[str, Any] = {"required_m": x_req, "descent_mps": ctx.tol.descent_mps}
    if len(p) == 0:
        return Verdict(Invariant.wall_descent.value, NA, "never airborne", m)
    # Descent rate from the true positions themselves (not the reported velocity).
    vd = np.zeros(len(p))
    if len(p) > 1:
        dt = np.diff(t)
        rate = np.where(dt > 1e-3, -np.diff(p[:, 2]) / np.maximum(dt, 1e-3), 0.0)
        vd = np.r_[rate, rate[-1]]
    ground = np.array([surface_z(ctx.grid, float(q[0]), float(q[1])) or float(ctx.home[2])
                       for q in p])
    sel = np.flatnonzero((vd > ctx.tol.descent_mps) & (p[:, 2] - ground >= WALL_MIN_HEIGHT_M))
    m["descending_samples"] = int(len(sel))
    if len(sel) == 0:
        return Verdict(Invariant.wall_descent.value, PASS, "no descent above "
                       f"{WALL_MIN_HEIGHT_M:g} m", m)
    d = wall_distance(ctx.grid, p[sel], x_req + 2.0)
    k = int(np.argmin(d))
    m["min_wall_m"] = None if not np.isfinite(d[k]) else float(d[k])
    m["at_t"], m["at"] = float(t[sel[k]]), [round(float(v), 2) for v in p[sel[k]]]
    close = sel[d < x_req]
    if len(close):
        m["seconds_close"] = float(len(close) * np.median(np.diff(t)) if len(t) > 1 else 0.0)
        return Verdict(Invariant.wall_descent.value, FAIL,
                       f"descending at {vd[sel[k]]:.1f} m/s {d[k]:.2f} m from a wall (< {x_req:g} m) "
                       f"at t={t[sel[k]]:.1f} s ({p[sel[k]][0]:.1f}, {p[sel[k]][1]:.1f}, "
                       f"{p[sel[k]][2]:.1f})", m)
    near = (f"no wall within {x_req + 2.0:g} m" if m["min_wall_m"] is None
            else f"walls ≥ {m['min_wall_m']:.2f} m away")
    return Verdict(Invariant.wall_descent.value, PASS,
                   f"{near} whenever descending (≥ {x_req:g} m required)", m)


# ── navigation (SIM-2n) ──────────────────────────────────────────────────
NAV_SETTLE_S = 2.0           # after a fix comes back, the EKF converges before it is judged


def check_navigation(rec: Recording, ctx: Context) -> Verdict:
    """GNSS-free runs: no GNSS where there is none, the EKF origin is the Site's, the EKF
    estimate stays on the truth while VPS has a fix, and a spoofed GNSS is flagged in time.
    The estimate's error on optical flow alone is reported, not judged (drift budget)."""
    n = ctx.nav
    if n is None:
        return Verdict(Invariant.navigation.value, NA, "a GNSS run")
    m: dict[str, Any] = {"gnss": n.gnss.value, "gnss_fixes": n.gnss_fixes, "origin_ok": n.origin_ok,
                         **{k: n.summary.get(k) for k in ("vps_error_h_rms_m", "vps_error_h_max_m",
                                                          "measurements", "source_sets")}}
    problems: list[str] = []
    if n.gnss is GnssUse.none and n.gnss_fixes:
        problems.append(f"{n.gnss_fixes} GNSS fixes reported on a vehicle without GNSS")
    if n.origin_required and not n.origin_ok:
        problems.append("the EKF origin was never confirmed as the Site origin")
    sl = airborne_slice(rec)
    t, p, e = rec.times[sl], rec.positions[sl], rec.estimates[sl]
    paired = np.isfinite(e[:, 0])
    err = np.full(len(t), np.nan)
    err[paired] = np.linalg.norm(e[paired, :2] - p[paired, :2], axis=1)
    fixed = n.has_fix(t)            # fix changes are on the capture clock = the truth's
    # Settled: with a fix for NAV_SETTLE_S already (after take-off or a regained fix).
    settled = fixed.copy()
    for t0, fix in n.fix:
        if fix:
            settled &= ~((t >= t0) & (t < t0 + NAV_SETTLE_S))
    on_vps = paired & settled
    on_flow = paired & ~fixed
    if on_vps.any():
        ev = err[on_vps]
        m.update(est_error_vps_p95_m=float(np.percentile(ev, 95)), est_error_vps_max_m=float(ev.max()),
                 est_error_vps_rms_m=float(np.sqrt((ev ** 2).mean())), samples_vps=int(on_vps.sum()))
        if m["est_error_vps_p95_m"] > ctx.tol.nav_error_m:
            problems.append(f"EKF estimate {m['est_error_vps_p95_m']:.2f} m (p95) from the truth "
                            f"with a VPS fix (allowed {ctx.tol.nav_error_m:g} m)")
    elif paired.any():
        problems.append("no airborne sample with a VPS fix")
    if on_flow.any():
        m.update(est_error_flow_max_m=float(err[on_flow].max()), samples_flow=int(on_flow.sum()))
    if n.spoof_injected_t is not None:
        m["spoof_injected_t"], m["spoof_flagged_t"] = n.spoof_injected_t, n.spoof_t
        if n.spoof_t is None:
            problems.append("GNSS was spoofed and never flagged")
        else:
            m["spoof_detect_s"] = n.spoof_t - n.spoof_injected_t
            if m["spoof_detect_s"] < 0:
                problems.append(f"GNSS flagged as spoofed {-m['spoof_detect_s']:.1f} s before "
                                f"the spoof (a false alarm)")
            elif m["spoof_detect_s"] > ctx.tol.spoof_detect_s:
                problems.append(f"spoofed GNSS flagged after {m['spoof_detect_s']:.1f} s "
                                f"(allowed {ctx.tol.spoof_detect_s:g} s)")
    elif n.spoof_t is not None:
        problems.append(f"GNSS flagged as spoofed at t={n.spoof_t:.1f} s without a spoof")
    if problems:
        return Verdict(Invariant.navigation.value, FAIL, "; ".join(problems), m)
    parts = [f"gnss {n.gnss.value}" + (", no GNSS fix at all" if n.gnss is GnssUse.none else "")]
    if "est_error_vps_p95_m" in m:
        parts.append(f"EKF vs truth with VPS: p95 {m['est_error_vps_p95_m']:.2f} m, "
                     f"max {m['est_error_vps_max_m']:.2f} m")
    if "est_error_flow_max_m" in m:
        parts.append(f"on optical flow up to {m['est_error_flow_max_m']:.2f} m")
    if "spoof_detect_s" in m:
        parts.append(f"spoof flagged after {m['spoof_detect_s']:.1f} s")
    return Verdict(Invariant.navigation.value, PASS, "; ".join(parts), m)


# ── all of them ──────────────────────────────────────────────────────────
def actual_outcome(rec: Recording, ctx: Context) -> str:
    start = ctx.mission_start_t
    if start is None:
        return "not_started"
    if any(md not in FLIGHT_MODES_OF_A_MISSION for _, md in modes_after(rec, start)):
        return Outcome.abort.value
    v = check_completion(rec, ctx)
    return Outcome.complete.value if v.metrics.get("mission_completed") and v.metrics.get("disarmed") \
        else "incomplete"


def judge(rec: Recording, ctx: Context, preflight: Optional[dict] = None) -> list[Verdict]:
    """Every invariant the scenario checks, in a fixed order."""
    trigger_t = ctx.trigger_t
    if ctx.expect.outcome is Outcome.abort and ctx.expect.reason == "fence_breach":
        trigger_t = fence_breach_time(rec, ctx)
    out: list[Verdict] = []
    series = None
    checked = list(ctx.expect.checked)
    if ctx.nav is not None and ctx.expect.invariants is None and Invariant.navigation not in checked:
        checked.append(Invariant.navigation)
    for inv in checked:
        if inv is Invariant.clearance:
            series = clearance_series(rec, ctx)
            out.append(check_clearance(rec, ctx, series))
        elif inv is Invariant.fence:
            out.append(check_fence(rec, ctx))
        elif inv is Invariant.hover:
            out.append(check_hover(rec, ctx))
        elif inv is Invariant.heading:
            out.append(check_heading(rec, ctx))
        elif inv is Invariant.altitude_frame:
            out.append(check_altitude_frame(rec, ctx, preflight))
        elif inv is Invariant.failsafe:
            out.append(check_failsafe(rec, ctx, trigger_t))
        elif inv is Invariant.completion:
            out.append(check_completion(rec, ctx))
        elif inv is Invariant.wall_descent:
            out.append(check_wall_descent(rec, ctx))
        elif inv is Invariant.navigation:
            out.append(check_navigation(rec, ctx))
    return out


def expected_status(inv: str, expect: Expect) -> str:
    """What a checked invariant must come out as for the scenario to pass."""
    if expect.outcome is Outcome.violation and inv == expect.reason:
        return FAIL
    if expect.outcome is Outcome.abort and expect.reason == "fence_breach" and inv == Invariant.fence.value:
        return FAIL
    return PASS
