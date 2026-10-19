"""Companion emulator for GNSS-free runs (SIM-2n, ADR-0015 §5 stage 1).

On a real vehicle the companion computer localises the vehicle against the twin (VPS, layer A),
feeds the pose to EKF3 as ExternalNav and reacts when VPS fails (§2.5). Here VPS is a model:
the measurement is the simulator's TRUE pose (SIM_STATE) plus a VPS error model, so the
autopilot side — EKF3 source sets, failsafes, arming, the mission — is the real thing and the
camera side is not (SIM-3 replaces the model by VPS on rendered IMX290 frames).

    origin        SET_GPS_GLOBAL_ORIGIN = the Site origin, repeated until GPS_GLOBAL_ORIGIN
                  reads back exactly that (EKF3 ignores an origin sent before it initialised,
                  and keeps the first origin it gets — from GNSS, if GNSS is faster)
    measurement   ODOMETRY (frame LOCAL_FRD = NED from the EKF origin, child BODY_FRD) at
                  `rate_hz`, built from SIM_STATE captured `latency_ms` earlier: position
                  + white noise σ + Gauss-Markov bias, yaw + noise, velocity (body frame) + noise,
                  pose / velocity covariance = what the model really is (σ² + σ_bias²)
    no fix        none sent while a vps_loss fault is on, inside a no-fix zone, or never
                  (`nav.extnav: false`)
    VPS lost      (no fix for LOSS_CONFIRM_S, or no truth at all for STALL_S: nothing can be
                  sent) -> MAV_CMD_SET_EKF_SOURCE_SET 2 (optical flow);
                  a wall within `wall_guard_m` -> GUIDED, back away along the wall's normal to
                  `retreat_m`, then LAND; else on flow for `drift_budget_s`, then GUIDED stop
                  (back away first if a wall is close by then), then LAND. A fix again for
                  `regain_s` -> source set 1; the mission resumes only if it was not stopped
    GNSS monitor  (gnss: monitor) GPS_RAW_INT vs VPS: a disagreement above
                  `spoof_threshold_m` for SPOOF_CONFIRM samples -> GNSS flagged as spoofed; it
                  is never an EKF3 source in this profile, so it stays out of navigation

**The forward rangefinders are a stand-in.** The wall is found by casting three horizontal rays
(vehicle heading and ±30°) into the twin's occupancy grid from the TRUE position, with noise
— what 2-3 forward rangefinders would measure if the twin were the world. The back-off
velocity is computed in the body frame from those ranges alone (no map, no position), as the
real companion would. Until SIM-2 physics provides rng_* readings from the JSON backend, this
is a model of the sensor, not the sensor.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from aerosplat.geo.crs import EnuFrame, GeoPoint
from aerosplat.mission.export import enu_from_geodetic
from aerosplat.mission.models import ExtNavSpec, GnssUse
from aerosplat.sim.scenario import NavSim
from aerosplat.surface.models import OCCUPIED, UNKNOWN
from aerosplat.surface.occupancy import OccupancyGrid

MAV_FRAME_LOCAL_FRD, MAV_FRAME_BODY_FRD, MAV_FRAME_BODY_OFFSET_NED = 20, 12, 9
MAV_ESTIMATOR_TYPE_VISION = 2
CMD_SET_EKF_SOURCE_SET = 42007
# SET_POSITION_TARGET_LOCAL_NED: velocity + yaw RATE (0: hold the heading). Position, acceleration
# and yaw ignored. Without the yaw rate, GUIDED turns the nose into the direction of travel —
# backing off, it turned away from the wall and the forward rangefinders lost it (first SITL run).
VELOCITY_HOLD_YAW_MASK = 0b010111000111
LOSS_CONFIRM_S = 0.5        # no fix this long = VPS lost (ArduPilot's own timeout is 0.3 s)
STALL_S = 2.0               # no SIM_STATE truth this long (a link stall): treated as lost too
SPOOF_CONFIRM = 3           # consecutive GNSS fixes beyond the threshold
RANGEFINDER_BEARINGS_DEG = (-30.0, 0.0, 30.0)   # forward rangefinders, body frame
RANGEFINDER_MAX_M = 20.0
COMMAND_PERIOD_S = 0.2      # GUIDED velocity setpoints (Copter times them out after 3 s)
BRAKE_S = 1.0               # zero velocity after the back-off, before LAND
ORIGIN_RESEND_S = 0.5       # wall clock


# ── pure pieces (unit-tested) ────────────────────────────────────────────
def ned_from_enu(enu) -> tuple[float, float, float]:
    """Site ENU (relative to the origin) -> NED from the same origin."""
    return float(enu[1]), float(enu[0]), -float(enu[2])


def quaternion(roll: float, pitch: float, yaw: float) -> list[float]:
    """Body -> NED rotation as [w, x, y, z] (aerospace 3-2-1 Euler angles, radians)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body -> NED rotation matrix (3-2-1)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [-sp, sr * cp, cr * cp]])


class GaussMarkov:
    """First-order Gauss-Markov process per axis: stationary σ, correlation time τ."""

    def __init__(self, sigma: float, tau_s: float, rng: random.Random, dims: int = 2):
        self.sigma, self.tau, self.rng = sigma, tau_s, rng
        self.x = [rng.gauss(0.0, sigma) if sigma > 0 else 0.0 for _ in range(dims)]

    def step(self, dt: float) -> list[float]:
        if self.sigma <= 0 or dt <= 0:
            return list(self.x)
        a = math.exp(-dt / self.tau)
        q = self.sigma * math.sqrt(1.0 - a * a)
        self.x = [a * v + q * self.rng.gauss(0.0, 1.0) for v in self.x]
        return list(self.x)


def wall_from_ranges(bearings_deg, ranges_m) -> Optional[tuple[float, tuple[float, float]]]:
    """(perpendicular distance, unit normal from the wall towards the vehicle) in the body
    frame (x forward, y right), from forward rangefinder hits (inf / NaN: no hit). Two or more
    hits: the line through them; one: the beam itself. None: nothing in range."""
    pts = [(r * math.cos(math.radians(b)), r * math.sin(math.radians(b)))
           for b, r in zip(bearings_deg, ranges_m) if r is not None and math.isfinite(r)]
    if not pts:
        return None
    p = np.asarray(pts, float)
    if len(p) == 1:
        d = float(np.linalg.norm(p[0]))
        n = -p[0] / max(d, 1e-9)
        return d, (float(n[0]), float(n[1]))
    # Least-squares line: its direction is the principal axis of the hit points.
    mean = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - mean)
    u = vt[0]
    n = np.array([-u[1], u[0]])
    if float(n @ (-mean)) < 0:          # point from the wall to the vehicle (at the origin)
        n = -n
    return float(abs(n @ mean)), (float(n[0]), float(n[1]))


def in_zone(xy, zones) -> bool:
    return any(math.hypot(xy[0] - z.center[0], xy[1] - z.center[1]) <= z.radius_m for z in zones)


@dataclass
class Truth:
    t: float
    enu: np.ndarray
    vel_ned: tuple[float, float, float]
    roll: float
    pitch: float
    yaw: float


@dataclass
class NavLog:
    """What the companion did, for the oracles and the report (simulation seconds)."""

    origin_sent: int = 0
    origin_confirmed_t: Optional[float] = None
    origin_readback: Optional[dict] = None
    measurements: int = 0
    meas_err_h: list[float] = field(default_factory=list)       # VPS fix - truth, horizontal
    fix: list[tuple[float, bool]] = field(default_factory=list)   # (t, has a fix) on change
    source_sets: list[tuple[float, int, int]] = field(default_factory=list)  # (t, set, MAV_RESULT)
    reactions: list[tuple[float, str, str]] = field(default_factory=list)    # (t, what, detail)
    walls: list[tuple[float, float]] = field(default_factory=list)          # (t, distance)
    gnss_fixes: int = 0                                         # GPS_RAW_INT with fix >= 2D
    gnss_err_h: list[tuple[float, float]] = field(default_factory=list)   # (t, |GNSS - VPS|)
    spoof_t: Optional[float] = None


class Companion:
    """Attach to a Vehicle; call `tick()` from the runner's loop."""

    def __init__(self, v: Any, *, origin: GeoPoint, sim: NavSim, spec: ExtNavSpec, flow: bool,
                 grid: OccupancyGrid, speedup: float, event: Callable[..., Any] = lambda *a, **k: None,
                 log: Callable[[str], Any] = lambda s: None):
        self.v, self.origin, self.sim, self.spec, self.flow = v, origin, sim, spec, flow
        self.grid, self.speedup, self.event, self.log = grid, speedup, event, log
        self.frame = EnuFrame(origin)
        self.rng = random.Random(sim.seed)
        self.sigma = sim.sigma_m if sim.sigma_m is not None else spec.sigma_m
        self.sigma_z = sim.sigma_z_m if sim.sigma_z_m is not None else spec.sigma_z_m
        self.yaw_sigma = math.radians(sim.yaw_sigma_deg if sim.yaw_sigma_deg is not None
                                      else spec.yaw_sigma_deg)
        self.vel_sigma = sim.vel_sigma_mps if sim.vel_sigma_mps is not None else spec.vel_sigma_mps
        self.latency = (sim.latency_ms if sim.latency_ms is not None else spec.latency_ms) / 1000.0
        self.period = 1.0 / (sim.rate_hz if sim.rate_hz is not None else spec.rate_hz)
        self.bias = GaussMarkov(sim.bias_sigma_m, sim.bias_tau_s, self.rng)
        self.nav = NavLog()
        self.vps_on = True                      # vps_loss fault (runner) switches this
        self.buffer: list[Truth] = []
        self.truth: Optional[Truth] = None
        self.last_capture = -math.inf
        self.last_bias_t: Optional[float] = None
        self.last_fix_t: Optional[float] = None
        self.has_fix: Optional[bool] = None
        self.fix_since: Optional[float] = None
        self.lost_t: Optional[float] = None
        self.source = 1
        self.phase = "nominal"          # nominal / flow / backoff / brake / land
        self.phase_t = 0.0
        self.last_cmd = -math.inf
        self.backoff_dir: Optional[tuple[float, float]] = None
        self.backoff_from_m = 0.0
        self.armed_since: Optional[float] = None
        self.origin_ok = False
        self._origin_sent_wall = -math.inf
        self._spoof_run = 0
        self.last_vps_ned: Optional[tuple[float, float]] = None

    # ── wiring ───────────────────────────────────────────────────────────
    def attach(self) -> None:
        self.v.listeners.append(self.on)

    def detach(self) -> None:
        if self.on in self.v.listeners:
            self.v.listeners.remove(self.on)

    def now(self) -> float:
        return self.v.sim_now(self.speedup)

    def on(self, msg: Any) -> None:
        kind = msg.get_type()
        if kind == "SIM_STATE":
            self._truth(msg)
        elif kind == "HEARTBEAT":
            self._send_origin()
        elif kind == "GPS_GLOBAL_ORIGIN":
            self._origin_readback(msg)
        elif kind == "GPS_RAW_INT":
            self._gnss(msg)

    # ── origin ───────────────────────────────────────────────────────────
    def origin_target(self) -> tuple[int, int, int]:
        """(lat 1e7, lon 1e7, alt mm) as sent and as it must read back."""
        return (int(round(self.origin.lat * 1e7)), int(round(self.origin.lon * 1e7)),
                int(round(self.origin.h * 1000.0)))

    def _send_origin(self) -> None:
        if not self.sim.origin or self.origin_ok or self.v.target_system == 0:
            return
        wall = time.monotonic()
        if wall - self._origin_sent_wall < ORIGIN_RESEND_S:
            return
        self._origin_sent_wall = wall
        lat, lon, alt = self.origin_target()
        self.v.conn.mav.set_gps_global_origin_send(self.v.target_system, lat, lon, alt, 0)
        self.nav.origin_sent += 1

    def _origin_readback(self, msg: Any) -> None:
        got = (int(msg.latitude), int(msg.longitude), int(msg.altitude))
        self.nav.origin_readback = {"lat": got[0] * 1e-7, "lon": got[1] * 1e-7, "alt": got[2] / 1000.0}
        want = self.origin_target()
        # 1e-7° exactly; altitude: int32 cm inside ArduPilot (mm -> cm -> mm), so ±10 mm.
        ok = got[0] == want[0] and got[1] == want[1] and abs(got[2] - want[2]) <= 10
        if ok and not self.origin_ok:
            self.origin_ok = True
            try:
                t = self.now()
            except Exception:
                t = 0.0
            self.nav.origin_confirmed_t = t
            self.event(t, "nav", "origin set", **self.nav.origin_readback)
        elif not ok:
            self.origin_ok = False

    def origin_mismatch(self) -> Optional[str]:
        """The EKF holds an origin, and it is not the Site origin (set by GNSS first?)."""
        rb = self.nav.origin_readback
        if rb is None or self.origin_ok:
            return None
        return (f"EKF origin {rb['lat']:.7f}, {rb['lon']:.7f}, {rb['alt']:.2f} m is not the Site "
                f"origin {self.origin.lat:.7f}, {self.origin.lon:.7f}, {self.origin.h:.2f} m")

    # ── measurements ─────────────────────────────────────────────────────
    def _truth(self, msg: Any) -> None:
        lat_i, lon_i = getattr(msg, "lat_int", 0), getattr(msg, "lon_int", 0)
        if lat_i == 0 and lon_i == 0:
            return
        try:
            t = self.now()
        except Exception:
            return
        enu = enu_from_geodetic(self.frame, lat_i * 1e-7, lon_i * 1e-7, float(msg.alt))[0]
        tr = Truth(t=t, enu=np.asarray(enu, float), vel_ned=(float(msg.vn), float(msg.ve), float(msg.vd)),
                   roll=float(msg.roll), pitch=float(msg.pitch), yaw=float(msg.yaw))
        self.truth = tr
        self.buffer.append(tr)
        # Captures whose latency has passed, at most one per period (the newest such one).
        due = [b for b in self.buffer if b.t <= t - self.latency]
        if not due:
            return
        self.buffer = [b for b in self.buffer if b.t > t - self.latency]
        cap = due[-1]
        if cap.t - self.last_capture < self.period - 1e-6:
            return
        self.last_capture = cap.t
        fix = self.fix_available(cap)
        self._fix_state(cap.t, fix)
        if fix:
            self._send(cap)

    def fix_available(self, cap: Truth) -> bool:
        return self.sim.extnav and self.vps_on and not in_zone(cap.enu[:2], self.sim.zones)

    def _fix_state(self, t: float, fix: bool) -> None:
        if fix:
            self.last_fix_t = t
        if fix != self.has_fix:
            self.has_fix = fix
            self.nav.fix.append((t, fix))
            self.fix_since = t
            self.event(t, "nav", "VPS fix" if fix else "VPS no fix")

    def _send(self, cap: Truth) -> None:
        dt = 0.0 if self.last_bias_t is None else cap.t - self.last_bias_t
        self.last_bias_t = cap.t
        bx, by = self.bias.step(dt)
        n, e, d = ned_from_enu(cap.enu)
        g = self.rng.gauss
        n += by + g(0.0, self.sigma)        # bias kept in ENU (x east, y north)
        e += bx + g(0.0, self.sigma)
        d += g(0.0, self.sigma_z)
        yaw = cap.yaw + g(0.0, self.yaw_sigma)
        q = quaternion(cap.roll, cap.pitch, yaw)
        v_ned = np.asarray(cap.vel_ned) + np.array([g(0.0, self.vel_sigma) for _ in range(3)])
        v_body = rotation(cap.roll, cap.pitch, yaw).T @ v_ned
        sh2 = self.sigma ** 2 + self.sim.bias_sigma_m ** 2
        pose_cov = [0.0] * 21
        pose_cov[0] = pose_cov[6] = sh2
        pose_cov[11] = self.sigma_z ** 2
        pose_cov[15] = pose_cov[18] = math.radians(1.0) ** 2
        pose_cov[20] = self.yaw_sigma ** 2
        vel_cov = [0.0] * 21
        vel_cov[0] = vel_cov[6] = vel_cov[11] = max(self.vel_sigma, 1e-3) ** 2
        nan = float("nan")
        self.v.conn.mav.odometry_send(int(cap.t * 1e6), MAV_FRAME_LOCAL_FRD, MAV_FRAME_BODY_FRD,
                                      n, e, d, q, float(v_body[0]), float(v_body[1]),
                                      float(v_body[2]), nan, nan, nan, pose_cov, vel_cov, 0,
                                      MAV_ESTIMATOR_TYPE_VISION, 100)
        self.nav.measurements += 1
        tn, te, _ = ned_from_enu(cap.enu)
        self.nav.meas_err_h.append(math.hypot(n - tn, e - te))
        self.last_vps_ned = (n, e)

    # ── GNSS monitor ─────────────────────────────────────────────────────
    def _gnss(self, msg: Any) -> None:
        if int(msg.fix_type) < 2:
            return
        self.nav.gnss_fixes += 1
        if self.sim.gnss is not GnssUse.monitor or self.last_vps_ned is None:
            return
        enu = enu_from_geodetic(self.frame, msg.lat * 1e-7, msg.lon * 1e-7, msg.alt / 1000.0)[0]
        gn, ge, _ = ned_from_enu(enu)
        err = math.hypot(gn - self.last_vps_ned[0], ge - self.last_vps_ned[1])
        try:
            t = self.now()
        except Exception:
            return
        self.nav.gnss_err_h.append((t, err))
        if err > self.sim.spoof_threshold_m:
            self._spoof_run += 1
            if self._spoof_run >= SPOOF_CONFIRM and self.nav.spoof_t is None:
                self.nav.spoof_t = t
                self.event(t, "nav", "GNSS rejected: disagrees with VPS", error_m=err)
                self.log(f"t={t:.1f} s GNSS disagrees with VPS by {err:.1f} m: flagged as spoofed")
        else:
            self._spoof_run = 0

    # ── reactions ────────────────────────────────────────────────────────
    def rangefinders(self) -> list[float]:
        """Forward ranges (m) from the TRUE pose into the twin; inf where nothing is in range."""
        tr = self.truth
        if tr is None:
            return [math.inf] * len(RANGEFINDER_BEARINGS_DEG)
        yaw_enu = math.pi / 2 - tr.yaw         # NED heading -> ENU angle from +x
        dirs = [(math.cos(yaw_enu - math.radians(b)), math.sin(yaw_enu - math.radians(b)), 0.0)
                for b in RANGEFINDER_BEARINGS_DEG]
        hits = self.grid.raycast(np.repeat(tr.enu[None], len(dirs), axis=0), np.asarray(dirs),
                                 RANGEFINDER_MAX_M)
        out = []
        for d, st in zip(hits.distance, hits.state):
            wall = math.isfinite(d) and int(st) in (OCCUPIED, UNKNOWN)
            out.append(max(0.0, float(d) + self.rng.gauss(0.0, self.sim.rangefinder_sigma_m))
                       if wall else math.inf)
        return out

    def wall(self) -> Optional[tuple[float, tuple[float, float]]]:
        return wall_from_ranges(RANGEFINDER_BEARINGS_DEG, self.rangefinders())

    def set_source(self, t: float, n: int) -> None:
        result = self.v.command(CMD_SET_EKF_SOURCE_SET, float(n))
        self.nav.source_sets.append((t, n, result))
        self.source = n
        self.event(t, "nav", f"EKF3 source set {n}", result=result)
        self.log(f"t={t:.1f} s EKF3 source set {n} (MAV_RESULT {result})")

    def _react(self, t: float, what: str, detail: str = "") -> None:
        self.nav.reactions.append((t, what, detail))
        self.event(t, "nav", what + (f": {detail}" if detail else ""))
        self.log(f"t={t:.1f} s companion: {what} {detail}".rstrip())

    def tick(self) -> None:
        """Run the VPS-loss reactions (ADR-0015 §2.5); call often from the runner's loop."""
        if not self.sim.extnav or self.truth is None:
            return
        t = self.now()
        r = self.sim.reaction
        if not self.v.armed:
            self.armed_since = None
            return
        if self.armed_since is None:
            self.armed_since = t
        # Lost: VPS reported no fix for LOSS_CONFIRM_S (on the capture clock), or nothing was
        # captured at all for STALL_S (the simulator's truth stopped: nothing can be sent).
        last_cap = self.last_capture if math.isfinite(self.last_capture) else self.armed_since
        stalled = t - max(last_cap, self.armed_since) > STALL_S + self.latency
        nofix = self.has_fix is False and self.fix_since is not None \
            and last_cap - self.fix_since >= LOSS_CONFIRM_S
        lost = nofix or stalled
        regained = (self.has_fix is True and self.fix_since is not None and not stalled
                    and last_cap - self.fix_since >= r.regain_s)
        if lost and self.lost_t is None and stalled and not nofix:
            self._react(t, "no truth from the simulator", f"{t - last_cap:.1f} s: no measurement sent")
        if lost and self.lost_t is None:
            self.lost_t = t
            self._react(t, "VPS lost")
            if self.flow and self.source != 2:
                self.set_source(t, 2)
            if self.phase not in ("nominal", "flow"):
                return                  # already stopping: the reaction under way goes on
            w = self.wall()
            if w is not None and w[0] < r.wall_guard_m:
                self._start_backoff(t, w)
            elif self.phase == "nominal":
                self.phase, self.phase_t = ("flow", t) if self.flow else ("brake", t)
                if not self.flow:
                    self._stop(t, "no optical flow: stop and land")
            return
        if self.lost_t is not None and regained:
            if self.source != 1:
                self.set_source(t, 1)
            self._react(t, "VPS regained", f"after {t - self.lost_t:.1f} s")
            self.lost_t = None
            if self.phase == "flow":
                self.phase = "nominal"
            return
        if self.phase == "flow":
            w = self.wall()
            if w is not None and w[0] < r.wall_guard_m:
                self._start_backoff(t, w)
            elif t - self.phase_t >= r.drift_budget_s:
                self._stop(t, f"drift budget {r.drift_budget_s:g} s on optical flow spent")
        elif self.phase == "backoff":
            self._backoff(t)
        elif self.phase == "brake":
            if t - self.last_cmd >= COMMAND_PERIOD_S:
                self._velocity(0.0, 0.0)
                self.last_cmd = t
            if t - self.phase_t >= BRAKE_S:
                w = self.wall()
                if w is not None and w[0] < r.retreat_m and self.backoff_dir is None:
                    self._start_backoff(t, w)
                else:
                    self._land(t)

    def _guided(self) -> None:
        if self.v.mode != "GUIDED":
            self.v.set_mode("GUIDED")

    def _stop(self, t: float, why: str) -> None:
        if self.v.mode == "LAND":       # already coming down where it is: let it (no wall near)
            self._react(t, "already landing", why)
            self.phase, self.phase_t = "land", t
            return
        self._react(t, "stop (GUIDED)", why)
        self._guided()
        self.phase, self.phase_t = "brake", t
        self._velocity(0.0, 0.0)
        self.last_cmd = t

    def _start_backoff(self, t: float, w: tuple[float, tuple[float, float]]) -> None:
        dist, n = w
        self.backoff_dir = n
        self._react(t, "back off from the wall",
                    f"{dist:.1f} m ahead (rangefinders), normal {n[0]:+.2f}, {n[1]:+.2f} (body)")
        self.nav.walls.append((t, dist))
        self.backoff_from_m = dist
        self._guided()
        self.phase, self.phase_t = "backoff", t
        self._backoff(t)

    def _backoff(self, t: float) -> None:
        r = self.sim.reaction
        w = self.wall()
        dist = math.inf if w is None else w[0]
        if t - self.last_cmd >= COMMAND_PERIOD_S:
            self.nav.walls.append((t, dist))
            self.last_cmd = t
            # Clear: the wall measured at the retreat distance — or out of the rangefinders'
            # view, but only once the commanded travel would have covered the distance (a wall
            # that merely left the beams is not "clear").
            travelled = (t - self.phase_t) * r.backoff_speed_mps
            clear = dist >= r.retreat_m if math.isfinite(dist) else \
                travelled >= r.retreat_m - self.backoff_from_m
            if clear or t - self.phase_t > r.backoff_timeout_s:
                self._react(t, "clear of the wall" if clear else "back-off timed out",
                            "nothing in range" if not math.isfinite(dist) else f"{dist:.1f} m")
                self.phase, self.phase_t = "brake", t
                self._velocity(0.0, 0.0)
                return
            # Re-estimate the normal from the current ranges while the wall is in view.
            n = w[1] if w is not None else self.backoff_dir
            assert n is not None
            self.backoff_dir = n
            self._velocity(n[0] * r.backoff_speed_mps, n[1] * r.backoff_speed_mps)

    def _velocity(self, fwd: float, right: float) -> None:
        self.v.conn.mav.set_position_target_local_ned_send(
            0, self.v.target_system, self.v.target_component, MAV_FRAME_BODY_OFFSET_NED,
            VELOCITY_HOLD_YAW_MASK, 0, 0, 0, fwd, right, 0, 0, 0, 0, 0, 0)

    def _land(self, t: float) -> None:
        self._react(t, "land", "clear of walls" if self.backoff_dir is not None else "")
        self.phase, self.phase_t = "land", t
        self.v.set_mode("LAND")

    def summary(self) -> dict:
        err = np.asarray(self.nav.meas_err_h, float)
        return {"measurements": self.nav.measurements,
                "vps_error_h_rms_m": float(np.sqrt((err ** 2).mean())) if len(err) else None,
                "vps_error_h_max_m": float(err.max()) if len(err) else None,
                "declared_sigma_m": self.spec.sigma_m, "model_sigma_m": self.sigma,
                "model_bias_sigma_m": self.sim.bias_sigma_m, "latency_s": self.latency,
                "period_s": self.period, "origin_sent": self.nav.origin_sent,
                "origin_confirmed_t": self.nav.origin_confirmed_t,
                "origin_readback": self.nav.origin_readback,
                "source_sets": [list(s) for s in self.nav.source_sets],
                "reactions": [list(r) for r in self.nav.reactions],
                "fix_changes": [list(f) for f in self.nav.fix],
                "walls": [list(w) for w in self.nav.walls[-50:]],
                "gnss_fixes": self.nav.gnss_fixes, "spoof_t": self.nav.spoof_t,
                "gnss_vps_max_m": max((e for _, e in self.nav.gnss_err_h), default=None)}
