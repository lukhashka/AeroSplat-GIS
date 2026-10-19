"""Pre-flight VPS for one mission (ADR-0015 §2.4, §5 stage 2): the corridor atlas and the
localizability of the route, from the same renders. Ground station; renderer-agnostic (the twin
through `aerosplat.vps.twin.EngineRenderer`, a synthetic scene in the tests).

    atlas, st = build_route_atlas(renderer, plan, cam, extractor, meta)   # AtlasLayout
    track = route_poses(plan, max_dt_s=1.0, hold_step_s=1.0)     # sampled in time
    report = localizability(renderer, atlas, track, cam, drift_budget_s=5)
    plan = attach(plan, atlas_hash, "vps/atlas.npz", report, ...)   # + `vps_gap` findings

**The route.** Home on the ground, the take-off column, every positional item of the plan in
order (holds where the item holds), down to the touchdown point. The vehicle's heading along it
is what ArduCopter does with the package: the take-off column at the take-off heading (the
vehicle is placed facing it, `TAKEOFF_HEADING_DEG`), each leg facing the next waypoint
(WP_YAW_BEHAVIOR 2), the approach and the hold at the target's heading (CONDITION_YAW). Times
come from a trapezoidal speed profile that stops at every waypoint with an acceleration *below*
what Copter uses (`ACCEL_MPS2`): durations are overestimated, never under — a gap's duration is
compared with a drift budget.

**The atlas.** Keyframes every `spacing_m` along the route, each at `yaws` headings around the
planned one (the heading in flight differs: look-ahead, wind, a reaction), rendered with the
vehicle profile's VPS camera and mount. Keyframes near the ground have too few points with depth
and drop out by themselves (`build_atlas`: min_points).

**Localizability.** Query frames between the keyframes (half a spacing along the route, heading
jitter, camera effects `perturb` level `level`), localized with a prior = truth + noise of
`prior_sigma_m` / `prior_yaw_deg` — what the companion will do in flight, on the same renders.
Per segment: fix rate, σ_h and error percentiles, false fixes; **gaps**: runs of queries without
a fix, from the last fix before to the first fix after (conservative), in metres and seconds:

    take-off gap   lift-off to the first fix (flown on optical flow)        > budget: blocking
    interior gap   between two fixes                                       > budget: blocking
    landing gap    last fix to touchdown, the part inside the landing column: reported (the
                   final descent onto the surveyed take-off spot is flown on flow, by design);
                   the part before the landing column counts as an interior gap

A blocking gap is a `vps_gap` finding (where, how long): the mission may not fly (fail-safe).

**What it does not show.** The queries are rendered from the same splat as the atlas: this is
the ceiling of the method on this twin, not a real camera against it (ADR-0015 §4, §8).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from aerosplat.vps import perturb
from aerosplat.vps.atlas import Atlas
from aerosplat.vps.build import DepthFilter, Renderer, View, build_atlas
from aerosplat.vps.features import Extractor, require_cv2
from aerosplat.vps.geometry import Intrinsics, angles, camera_rotation, pose_error
from aerosplat.vps.localize import Localizer, PosePrior

NAV_TAKEOFF, NAV_WAYPOINT, NAV_LAND = 22, 16, 21
DO_CHANGE_SPEED, CONDITION_YAW = 178, 115
TAKEOFF_HEADING_DEG = 0.0       # the vehicle is placed on the take-off spot facing north
ACCEL_MPS2 = 1.0                # below Copter's WP accelerations: durations err on the long side
FALSE_FIX_M = 2.0
SEGMENTS = ("takeoff", "transit", "descent", "approach", "hover", "exit", "return", "landing")


@dataclass(frozen=True)
class RoutePose:
    s: float                        # metres along the route (3D path length)
    t: float                        # planned seconds from lift-off
    segment: str
    C: tuple[float, float, float]   # vehicle position, Site ENU
    yaw_deg: float                  # vehicle heading (compass)
    height_m: float                 # above home


@dataclass(frozen=True)
class CameraModel:
    """The VPS camera as the companion runs it: intrinsics at the processing resolution
    (OpenCV convention) and the mount on the vehicle."""

    width: int
    height: int
    K: Intrinsics
    mount_pitch_deg: float
    mount_yaw_deg: float = 0.0
    name: str = ""

    @classmethod
    def of(cls, camera: Any, mount: Any, resolution: tuple[int, int]) -> "CameraModel":
        """From an `aerosplat.sim.camera.Camera` and a vehicle profile's CameraMount."""
        w, h = resolution
        s = w / camera.width
        if abs(h / camera.height - s) > 1e-6:
            raise ValueError(f"{w}x{h} is not the {camera.width}x{camera.height} sensor scaled")
        return cls(width=w, height=h, K=(camera.fx * s, camera.fy * s, w / 2 - 0.5, h / 2 - 0.5),
                   mount_pitch_deg=float(mount.pitch_deg), mount_yaw_deg=float(mount.yaw_deg),
                   name=camera.name)

    def view(self, C: Sequence[float], roll: float, pitch: float, yaw: float) -> View:
        """The camera's view for a vehicle at C (ENU) with this attitude (radians, NED)."""
        y, p, r = angles(camera_rotation(roll, pitch, yaw, self.mount_pitch_deg, self.mount_yaw_deg))
        return View((float(C[0]), float(C[1]), float(C[2])), y, p, r)


# ── the planned flight ───────────────────────────────────────────────────
def _heading(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    d = b[:2] - a[:2]
    if float(np.hypot(*d)) < 0.5:
        return None
    return math.degrees(math.atan2(d[0], d[1])) % 360.0


def _leg_time(dist: float, v: float, a: float = ACCEL_MPS2) -> float:
    """Trapezoidal profile from rest to rest."""
    if dist <= 0:
        return 0.0
    d_acc = v * v / a
    if dist <= d_acc:
        return 2.0 * math.sqrt(dist / a)
    return 2.0 * v / a + (dist - d_acc) / v


def _dist_at(tau: float, dist: float, v: float, a: float = ACCEL_MPS2) -> float:
    """Distance covered after `tau` seconds of the same profile."""
    total = _leg_time(dist, v, a)
    tau = min(max(tau, 0.0), total)
    if dist <= v * v / a:
        half = total / 2.0
        return 0.5 * a * tau * tau if tau <= half else dist - 0.5 * a * (total - tau) ** 2
    t_a = v / a
    if tau <= t_a:
        return 0.5 * a * tau * tau
    if tau <= total - t_a:
        return 0.5 * v * t_a + v * (tau - t_a)
    return dist - 0.5 * a * (total - tau) ** 2


def _segment(role_to: str, role_from: str, dz: float, horizontal: float) -> str:
    if role_to == "takeoff":
        return "takeoff"
    if role_to == "land":
        return "landing"
    if role_to == "hover":
        return "approach"
    if role_from == "hover":
        return "exit"
    if horizontal < 0.5 and dz < 0:
        return "descent"
    return "transit" if role_to in ("over_entry", "entry") else "return"


def route_poses(plan: Any, spacing_m: float = 4.0, max_dt_s: float = math.inf,
                hold_step_s: float = 5.0) -> list[RoutePose]:
    """The planned flight of `plan` (a MissionPlan) sampled at least every `spacing_m` and every
    `max_dt_s` of planned flight, plus one pose every `hold_step_s` while holding (segment
    `hover` at the hover item). Keyframes: spatial spacing only; localizability queries: a time
    step too, so that a slow approach or a hold is sampled as densely as a transit."""
    if spacing_m <= 0 or max_dt_s <= 0 or hold_step_s <= 0:
        raise ValueError("spacing_m, max_dt_s and hold_step_s must be positive")
    prof = plan.profile
    home = np.asarray(plan.home.enu, float)
    speed = prof.transit_speed_mps
    yaw = TAKEOFF_HEADING_DEG
    fixed_yaw: Optional[float] = None
    out: list[RoutePose] = []
    s = t = 0.0
    pos = home.copy()
    prev_role = "home"

    def add(p: np.ndarray, seg: str, hdg: float) -> None:
        out.append(RoutePose(s=round(s, 3), t=round(t, 3), segment=seg,
                             C=(float(p[0]), float(p[1]), float(p[2])), yaw_deg=hdg % 360.0,
                             height_m=float(p[2] - home[2])))

    add(pos, "takeoff", yaw)
    for it in plan.items:
        if it.command == DO_CHANGE_SPEED and it.param2 > 0:
            speed = float(it.param2)
            continue
        if it.command == CONDITION_YAW:
            fixed_yaw = float(it.param1)
            continue
        if it.enu is None or it.command not in (NAV_TAKEOFF, NAV_WAYPOINT, NAV_LAND):
            continue
        to = np.asarray(it.enu, float)
        d = to - pos
        dist = float(np.linalg.norm(d))
        horiz = float(np.hypot(d[0], d[1]))
        seg = _segment(it.role, prev_role, float(d[2]), horiz)
        if it.command == NAV_TAKEOFF:
            v = prof.climb_speed_mps
        elif it.command == NAV_LAND or (horiz < 0.5 and d[2] < 0):
            v = prof.descent_speed_mps
        elif horiz < 0.5 and d[2] > 0:
            v = prof.climb_speed_mps
        else:
            v = speed
        h = _heading(pos, to)
        if fixed_yaw is not None:
            yaw = fixed_yaw
        elif h is not None:
            yaw = h
        t_leg = _leg_time(dist, v)
        # time-uniform samples; n bounds the distance step (by the peak speed) and the time step
        v_peak = min(v, math.sqrt(ACCEL_MPS2 * dist))
        n = max(1, int(math.ceil(v_peak * t_leg / spacing_m)) if dist > 0 else 1,
                int(math.ceil(t_leg / max_dt_s)) if math.isfinite(max_dt_s) else 1)
        for k in range(1, n + 1):
            tau = t_leg * k / n
            x = _dist_at(tau, dist, v) if dist > 0 else 0.0
            p = pos + (x / dist if dist > 0 else 1.0) * d
            out.append(RoutePose(s=round(s + x, 3), t=round(t + tau, 3), segment=seg,
                                 C=(float(p[0]), float(p[1]), float(p[2])), yaw_deg=yaw % 360.0,
                                 height_m=float(p[2] - home[2])))
        s += dist
        t += t_leg
        pos = to
        hold = float(it.param1) if it.command == NAV_WAYPOINT else 0.0
        if hold > 0:
            hseg = "hover" if it.role == "hover" else seg
            steps = max(1, int(math.ceil(hold / hold_step_s)))
            for k in range(1, steps + 1):
                t_h = t + hold * k / steps
                out.append(RoutePose(s=round(s, 3), t=round(t_h, 3), segment=hseg,
                                     C=(float(pos[0]), float(pos[1]), float(pos[2])),
                                     yaw_deg=yaw % 360.0, height_m=float(pos[2] - home[2])))
            t += hold
        if it.role == "hover":
            fixed_yaw = None                 # the next waypoint turns the nose (WP_YAW_BEHAVIOR)
        prev_role = it.role
    return out


# ── the atlas ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AtlasLayout:
    """Where the corridor atlas has keyframes. Along the route every `spacing_m`, at `yaws`
    around the planned heading. Near the target (`near_segments`) the scene is metres away, so a
    keyframe sees little of what a camera half a metre off sees: there every `near_spacing_m`,
    with lateral offsets across the heading (the corridor's σ) and finer headings. (Measured on
    dataset_1910: with the route-only layout, frames 0.5 m off the hover point 3 m from the
    facade were refused 70 % of the time.)"""

    spacing_m: float = 4.0
    yaws: tuple[float, ...] = (0, 45, -45, 90, -90, 180)
    near_segments: tuple[str, ...] = ("approach", "hover", "exit")
    near_spacing_m: float = 1.0
    near_lateral_m: tuple[float, ...] = (-0.5, 0.0, 0.5)
    near_yaws: tuple[float, ...] = (0, 8, -8, 16, -16, 30, -30, 180)
    hover_along_m: tuple[float, ...] = (-0.5, 0.0, 0.5)     # holding: towards / away from the target
    min_height_m: float = 1.0           # lower, the camera sees the ground at arm's length


def keyframe_views(plan: Any, cam: CameraModel, layout: AtlasLayout = AtlasLayout()) -> list[View]:
    """The keyframe views of `plan`'s corridor (deduplicated: holds repeat a pose)."""
    poses = route_poses(plan, spacing_m=min(layout.spacing_m, layout.near_spacing_m))
    seen: set[tuple] = set()
    out: list[View] = []
    last_s = -math.inf
    for p in poses:
        near = p.segment in layout.near_segments
        if p.height_m < layout.min_height_m:
            continue
        if not near:
            if p.s - last_s < layout.spacing_m - 1e-6:
                continue
            last_s = p.s
        yaw0 = math.radians(p.yaw_deg)
        across = np.array([math.cos(yaw0), -math.sin(yaw0), 0.0])     # to the right of the heading
        along = np.array([math.sin(yaw0), math.cos(yaw0), 0.0])
        offsets = [(a, b) for a in (layout.hover_along_m if p.segment == "hover" else (0.0,))
                   for b in (layout.near_lateral_m if near else (0.0,))]
        for fwd, lat in offsets:
            C = np.asarray(p.C, float) + lat * across + fwd * along
            for dy in (layout.near_yaws if near else layout.yaws):
                yaw = (p.yaw_deg + float(dy)) % 360.0
                key = (round(C[0], 2), round(C[1], 2), round(C[2], 2), round(yaw, 1))
                if key in seen:
                    continue
                seen.add(key)
                out.append(cam.view(C, 0.0, 0.0, math.radians(yaw)))
    return out


def build_route_atlas(renderer: Renderer, plan: Any, cam: CameraModel, extractor: Extractor,
                      meta: dict[str, Any], layout: AtlasLayout = AtlasLayout(),
                      log: Optional[Callable[[str], None]] = None) -> tuple[Atlas, dict[str, Any]]:
    views = keyframe_views(plan, cam, layout)
    if not views:
        raise ValueError("no route pose high enough for a keyframe")
    full = {**meta, "kind": "route", "mission": plan.id, "camera": cam.name,
            "mount_pitch_deg": cam.mount_pitch_deg, "mount_yaw_deg": cam.mount_yaw_deg,
            "layout": asdict(layout)}
    return build_atlas(renderer, views, cam.width, cam.height, cam.K, extractor, full, DepthFilter(),
                       log=log)


# ── localizability ───────────────────────────────────────────────────────
@dataclass
class QueryResult:
    s: float
    t: float
    segment: str
    height_m: float
    C: tuple[float, float, float]
    ok: bool
    reason: Optional[str]
    err_m: Optional[float] = None
    err_h_m: Optional[float] = None
    sigma_h_m: Optional[float] = None
    ms: Optional[float] = None


@dataclass
class Gap:
    kind: str                   # takeoff / interior / landing
    s0: float                   # route metres: last fix before (or lift-off)
    s1: float                   # first fix after (or touchdown)
    t0: float                   # planned seconds, same ends
    t1: float
    segments: list[str]         # segments of the queries without a fix
    at: tuple[float, float, float]   # where the first query without a fix was
    counted_s: float            # the part of it compared with the drift budget
    blocking: bool

    @property
    def length_m(self) -> float:
        return self.s1 - self.s0

    @property
    def duration_s(self) -> float:
        return self.t1 - self.t0

    def describe(self) -> str:
        return (f"no VPS fix for {self.duration_s:.1f} s / {self.length_m:.1f} m ({self.kind}, "
                f"{'/'.join(self.segments)}, route {self.s0:.1f}-{self.s1:.1f} m, "
                f"t {self.t0:.1f}-{self.t1:.1f} s)")


@dataclass
class Localizability:
    drift_budget_s: float
    level: str
    prior_sigma_m: float
    queries: list[QueryResult] = field(default_factory=list)
    segments: dict[str, dict[str, Any]] = field(default_factory=dict)
    gaps: list[Gap] = field(default_factory=list)
    false_fixes: int = 0

    @property
    def blocking(self) -> list[Gap]:
        return [g for g in self.gaps if g.blocking]

    @property
    def fix_rate(self) -> Optional[float]:
        return round(sum(q.ok for q in self.queries) / len(self.queries), 3) if self.queries else None

    def summary(self) -> dict[str, Any]:
        """Everything but the per-query rows (what the plan carries)."""
        return {"drift_budget_s": self.drift_budget_s, "level": self.level,
                "prior_sigma_m": self.prior_sigma_m, "false_fixes": self.false_fixes,
                "fix_rate": self.fix_rate, "queries": len(self.queries), "segments": self.segments,
                "gaps": [{**asdict(g), "length_m": round(g.length_m, 2),
                          "duration_s": round(g.duration_s, 2)} for g in self.gaps]}

    def as_dict(self) -> dict[str, Any]:
        return {**self.summary(), "rows": [asdict(q) for q in self.queries]}


def _pct(vals: list[float], q: float) -> Optional[float]:
    return None if not vals else round(float(np.percentile(vals, q)), 3)


def find_gaps(results: Sequence[QueryResult], track: Sequence[RoutePose],
              drift_budget_s: float) -> list[Gap]:
    """Runs of queries without a fix along the route (in planned time), each from the last fix
    before to the first fix after: the vehicle flies that whole stretch on layer B. Kinds and
    which of them block: see the module docstring. Deterministic, no rendering."""
    if not track:
        return []
    start, end = track[0], track[-1]
    # the landing column begins where the last non-landing pose is
    land_t0 = max((p.t for p in track if p.segment != "landing"), default=end.t)
    rs = sorted(results, key=lambda r: (r.t, r.s))
    out: list[Gap] = []
    k = 0
    while k < len(rs):
        if rs[k].ok:
            k += 1
            continue
        j = k
        while j < len(rs) and not rs[j].ok:
            j += 1
        before = rs[k - 1] if k > 0 else None
        after = rs[j] if j < len(rs) else None
        segs = list(dict.fromkeys(r.segment for r in rs[k:j]))
        at = rs[k].C
        if before is None and after is None:
            out.append(Gap("interior", start.s, end.s, start.t, end.t, segs, at,
                           end.t - start.t, True))
        elif before is None:
            assert after is not None
            span = after.t - start.t
            out.append(Gap("takeoff", start.s, after.s, start.t, after.t, segs, at, span,
                           span > drift_budget_s))
        elif after is None:
            pre = max(0.0, land_t0 - before.t)       # flown on flow before the landing column
            out.append(Gap("landing", before.s, end.s, before.t, end.t, segs, at, pre,
                           pre > drift_budget_s))
        else:
            span = after.t - before.t
            out.append(Gap("interior", before.s, after.s, before.t, after.t, segs, at, span,
                           span > drift_budget_s))
        k = j
    return out


# How far off the plan the vehicle is expected when a frame is taken: (1σ horizontal m, 1σ
# vertical m, heading ± uniform deg). In transit the corridor's scale; near the target what SITL
# showed while holding (SIM-1 / SIM-2n: 0.1-0.6 m, heading ± 2°) with margin.
DEVIATION = {"default": (0.5, 0.3, 10.0), "approach": (0.3, 0.2, 5.0), "hover": (0.3, 0.2, 5.0),
             "exit": (0.3, 0.2, 5.0)}


def query_poses(track: Sequence[RoutePose], rng: np.random.Generator
                ) -> list[tuple[RoutePose, float]]:
    """Where the localizability frames are taken, each with its heading jitter: halfway between
    consecutive track poses (at the pose itself while holding), moved by a random offset
    (`DEVIATION` of the segment) so that no query sits exactly on a keyframe."""
    out: list[tuple[RoutePose, float]] = []
    for a, b in zip(track, track[1:]):
        hold = a.C == b.C
        f = 1.0 if hold else 0.5
        C = np.asarray(a.C, float) + f * (np.asarray(b.C, float) - np.asarray(a.C, float))
        sh, sz, yaw_jitter = DEVIATION.get(b.segment, DEVIATION["default"])
        if b.height_m > 1.0:
            C = C + np.array([rng.normal(0, sh), rng.normal(0, sh), rng.normal(0, sz)])
        out.append((RoutePose(s=a.s + f * (b.s - a.s), t=a.t + f * (b.t - a.t), segment=b.segment,
                              C=(float(C[0]), float(C[1]), float(C[2])), yaw_deg=b.yaw_deg,
                              height_m=a.height_m + f * (b.height_m - a.height_m)), yaw_jitter))
    return out


def localizability(renderer: Renderer, atlas: Atlas, track: Sequence[RoutePose], cam: CameraModel,
                   *, drift_budget_s: float, level: str = "mild", prior_sigma_m: float = 3.0,
                   prior_yaw_deg: float = 5.0, seed: int = 0,
                   localizer: Optional[Localizer] = None,
                   log: Optional[Callable[[str], None]] = None) -> Localizability:
    """Localize one frame per `query_poses(track)` as the companion would (camera effects
    `level`, prior = truth + noise) and summarise per segment and in gaps. `track`: route poses
    sampled in time (`route_poses(plan, max_dt_s=1, hold_step_s=1)`). Deterministic for a
    deterministic renderer and a given seed."""
    cv2 = require_cv2()
    rng = np.random.default_rng(seed)
    loc = localizer or Localizer(atlas)
    rep = Localizability(drift_budget_s=drift_budget_s, level=level, prior_sigma_m=prior_sigma_m)
    queries = query_poses(track, rng)
    t0 = time.perf_counter()
    for i, (q, yaw_jitter_deg) in enumerate(queries):
        yaw = q.yaw_deg + float(rng.uniform(-yaw_jitter_deg, yaw_jitter_deg))
        roll, pitch = (math.radians(float(v)) for v in rng.uniform(-3, 3, 2))
        view = cam.view(q.C, roll, pitch, math.radians(yaw))
        rgb, _ = renderer.render(view, cam.width, cam.height, cam.K)
        img = perturb.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), perturb.draw(level, rng))
        prior_C = np.asarray(q.C, float) + np.array([rng.normal(0, prior_sigma_m),
                                                     rng.normal(0, prior_sigma_m), rng.normal(0, 1.0)])
        pv = cam.view(prior_C, roll, pitch, math.radians(yaw + float(rng.normal(0, prior_yaw_deg))))
        prior = PosePrior(C=pv.C, yaw_deg=pv.yaw_deg, pitch_deg=pv.pitch_deg, roll_deg=pv.roll_deg,
                          sigma_pos_m=prior_sigma_m, sigma_yaw_deg=prior_yaw_deg)
        fix = loc.localize(img, cam.K, prior)
        r = QueryResult(s=round(q.s, 3), t=round(q.t, 3), segment=q.segment,
                        height_m=round(q.height_m, 2), C=q.C, ok=fix.ok, reason=fix.reason,
                        ms=fix.timings_ms.get("total"))
        if fix.ok:
            assert fix.R is not None and fix.C is not None
            e = pose_error(fix.R, fix.C, view.R, np.asarray(view.C, float))
            r.err_m, r.err_h_m = round(e["pos_m"], 3), round(e["horiz_m"], 3)
            r.sigma_h_m = round(float(fix.sigma_h_m or 0.0), 3)
        rep.queries.append(r)
        if log is not None and (i + 1) % 25 == 0:
            log(f"   localizability {i + 1}/{len(queries)} ({time.perf_counter() - t0:.0f} s)")
    rep.false_fixes = sum(1 for r in rep.queries if r.ok and (r.err_m or 0.0) > FALSE_FIX_M)
    for seg in SEGMENTS:
        rows = [r for r in rep.queries if r.segment == seg]
        if not rows:
            continue
        ok = [r for r in rows if r.ok]
        reasons: dict[str, int] = {}
        for r in rows:
            if not r.ok and r.reason:
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        rep.segments[seg] = {
            "queries": len(rows), "fix_rate": round(len(ok) / len(rows), 3),
            "sigma_h_m_p50": _pct([r.sigma_h_m for r in ok if r.sigma_h_m is not None], 50),
            "sigma_h_m_p95": _pct([r.sigma_h_m for r in ok if r.sigma_h_m is not None], 95),
            "err_h_m_p95": _pct([r.err_h_m for r in ok if r.err_h_m is not None], 95),
            "false_fixes": sum(1 for r in ok if (r.err_m or 0.0) > FALSE_FIX_M),
            "refusals": dict(sorted(reasons.items())),
            "lowest_fix_height_m": min((r.height_m for r in ok), default=None)}
    rep.gaps = find_gaps(rep.queries, track, drift_budget_s)
    return rep


# ── into the plan ────────────────────────────────────────────────────────
ATLAS_FILE = "vps/atlas.npz"            # in the mission folder
ANCHOR_SIGMA_M, ANCHOR_SIGMA_YAW_DEG = 0.3, 3.0


def gap_findings(rep: Localizability) -> list[Any]:
    """Blocking gaps as `vps_gap` findings of the plan (where, how long)."""
    from aerosplat.mission.models import Code, Finding, Severity

    return [Finding(code=Code.vps_gap, severity=Severity.blocking, leg=g.kind,
                    detail=f"{g.describe()}: {g.counted_s:.1f} s on layer B is longer than its "
                           f"drift budget ({rep.drift_budget_s:g} s)", at=g.at)
            for g in rep.blocking]


def attach(plan: Any, atlas: Atlas, rep: Localizability, cam: CameraModel, *, run_id: str,
           file: str = ATLAS_FILE) -> Any:
    """`plan` (a GNSS-free MissionPlan) with its VPS binding (atlas by hash, localizability,
    take-off anchor) and the `vps_gap` findings; the verdict follows the findings. A plan that
    already has `vps_gap` findings loses them first (a re-attached atlas replaces the old)."""
    from aerosplat.mission.models import (Code, TakeoffAnchor, VpsAtlasRef, VpsBinding,
                                          verdict_of)

    if not plan.profile.nav.gnss_free:
        raise ValueError("a VPS atlas belongs to a GNSS-free plan (profile.nav.gnss none / monitor)")
    digest = atlas.meta.get("content_hash")
    if not digest or digest != atlas.content_hash():
        raise ValueError("save the atlas first: its stored hash must be its content hash")
    if atlas.meta.get("mission") not in (None, plan.id):
        raise ValueError(f"the atlas was built for mission {atlas.meta.get('mission')}, not {plan.id}")
    intr = atlas.meta["intrinsics"]
    ref = VpsAtlasRef(file=file, content_hash=digest, extractor=atlas.meta["extractor"]["name"],
                      resolution=(int(intr["width"]), int(intr["height"])), camera=cam.name,
                      mount_pitch_deg=cam.mount_pitch_deg, mount_yaw_deg=cam.mount_yaw_deg,
                      keyframes=atlas.n_keyframes, features=atlas.n_features, run_id=run_id,
                      splat_sha256=atlas.meta.get("splat_sha256"))
    anchor = TakeoffAnchor(enu=plan.home.enu, heading_deg=TAKEOFF_HEADING_DEG,
                           sigma_m=ANCHOR_SIGMA_M, sigma_yaw_deg=ANCHOR_SIGMA_YAW_DEG)
    findings = [f for f in plan.findings if f.code is not Code.vps_gap] + gap_findings(rep)
    return plan.model_copy(update={
        "vps": VpsBinding(atlas=ref, anchor=anchor, localizability=rep.summary()),
        "findings": findings, "verdict": verdict_of(findings)})


# ── the whole pre-flight step ────────────────────────────────────────────
DRIFT_BUDGET_S = 5.0            # layer B (optical flow) without a fix: the companion's default


def prepare(plan: Any, renderer: Renderer, cam: CameraModel, extractor: Extractor, folder: Path, *,
            run_id: str, meta: Optional[dict[str, Any]] = None, layout: AtlasLayout = AtlasLayout(),
            drift_budget_s: float = DRIFT_BUDGET_S, level: str = "mild", seed: int = 0,
            log: Optional[Callable[[str], None]] = None) -> tuple[Any, Atlas, Localizability]:
    """Atlas of `plan`'s corridor saved to `folder`/ATLAS_FILE, the route's localizability
    (`folder`/vps/localizability.json, with every query) and the plan bound to both."""
    t0 = time.perf_counter()
    atlas, stats = build_route_atlas(renderer, plan, cam, extractor, dict(meta or {}, run=run_id),
                                     layout, log=log)
    path = Path(folder) / ATLAS_FILE
    atlas.save(path)
    if log is not None:
        log(f"   atlas {atlas.n_keyframes} keyframes, {atlas.n_features} points, "
            f"{path.stat().st_size / 2 ** 20:.1f} MB, {time.perf_counter() - t0:.0f} s")
    track = route_poses(plan, spacing_m=layout.spacing_m, max_dt_s=1.0, hold_step_s=1.0)
    rep = localizability(renderer, atlas, track, cam, drift_budget_s=drift_budget_s, level=level,
                         seed=seed, log=log)
    out = attach(plan, atlas, rep, cam, run_id=run_id)
    (path.parent / "localizability.json").write_text(
        json.dumps({"atlas_hash": atlas.meta["content_hash"], "build": stats, **rep.as_dict()},
                   indent=1, default=float), encoding="utf-8")
    return out, atlas, rep


def main(argv: Optional[list[str]] = None) -> int:
    """python -m aerosplat.vps.route --run-dir <site run> --mission <export folder> --out <dir>

    Ground station (GPU): the mission's corridor atlas and localizability; writes a new package
    to --out (mission.json + .waypoints / .plan / .parm + vps/atlas.npz + vps/localizability.json).
    The input folder is only read. Exit 2: a blocking finding (`vps_gap`), nothing exported."""
    import argparse

    from aerosplat.logging_utils import fix_console_encoding
    from aerosplat.mission.export import render
    from aerosplat.mission.models import MissionPlan
    from aerosplat.sim.camera import load_camera
    from aerosplat.sim.runner import load_vehicle
    from aerosplat.vps.features import make_extractor
    from aerosplat.vps.twin import EngineRenderer, SiteRun

    fix_console_encoding()
    ap = argparse.ArgumentParser(prog="python -m aerosplat.vps.route", description=main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--mission", required=True, type=Path, help="export folder (mission.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--vehicle", default="x500-pi5-imx290")
    ap.add_argument("--drift-budget", type=float, default=DRIFT_BUDGET_S)
    ap.add_argument("--accept-risks", action="store_true", help="export a plan with risks")
    a = ap.parse_args(argv)
    if a.out.resolve() == a.mission.resolve():
        raise SystemExit("--out must be another folder: the input package is only read")
    plan = MissionPlan(**json.loads((a.mission / "mission.json").read_text(encoding="utf-8")))
    run = SiteRun.open(a.run_dir)
    veh = load_vehicle(a.vehicle)
    mount = veh.airframe.vps_mount()
    if mount is None or mount.vps is None:
        raise SystemExit(f"vehicle {a.vehicle} has no VPS camera")
    cam = CameraModel.of(load_camera(mount.camera), mount, tuple(mount.vps.resolution))
    renderer = EngineRenderer(run)
    try:
        plan, atlas, rep = prepare(plan, renderer, cam, make_extractor(mount.vps.extractor),
                                   a.out, run_id=run.run_id,
                                   meta={"site": run.site_id, "splat": run.splat.name,
                                         "splat_sha256": run.splat_sha256()},
                                   drift_budget_s=a.drift_budget, log=print)
    finally:
        renderer.close()
    (a.out / "mission.json").write_text(plan.model_dump_json(indent=2, by_alias=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.summary().items() if k != "segments"}, indent=1,
                     default=float))
    for f in plan.blocking:
        print(f"BLOCKING {f.code.value}: {f.detail}")
    if plan.blocking:
        return 2
    for fmt in ("waypoints", "plan", "parm"):
        name, _, text = render(plan, run.origin, fmt, accept_risks=a.accept_risks)
        (a.out / name).write_text(text, encoding="utf-8")
    print(f"package {a.out} (atlas {atlas.meta['content_hash'][:12]})")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
