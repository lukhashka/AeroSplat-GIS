"""VPS in the loop (SIM-3, ADR-0015 §5 stage 2): the companion emulator localizes rendered camera
frames against the mission's atlas instead of adding an error model to the truth.

Every VPS tick (the vehicle profile's rate, 2-4 Hz):

    truth (SIM_STATE)  -> camera pose (vehicle attitude + mount) -> frame rendered in-process
                          (the site run's splat, 960x540 IMX290) -> camera effects: rolling-shutter
                          skew from the TRUE angular rate, then `perturb` (noise, blur, exposure)
    autopilot estimate -> the localizer's prior (GLOBAL_POSITION_INT + ATTITUDE: what the
                          vehicle believes, never the truth; σ grows while there is no fix)
    Localizer          -> a fix: ODOMETRY with the localizer's covariance, timestamped at the
                          capture; a refusal: nothing is sent (VPS loss, the SIM-2n reactions)

Truth places the camera and drives its rolling shutter; besides that it is only recorded (for
the `vps` oracle: the true error of every fix that was sent).

**Time.** SITL's built-in physics runs on its own clock and cannot be paused per frame, so a
VPS run flies at speedup 1: render + localize take real time, and a result goes out when it is
ready — compute latency is real latency. The timestamp is the capture time, and no result goes
out sooner than the declared latency (VISO_DELAY_MS) after its capture: ArduPilot maps companion
time to its own by the smallest lag it has seen (JitterCorrection) and then subtracts
VISO_DELAY_MS, so the declared latency must be the floor of the real one (the anchor is stamped
the same way). The render (~5-20 ms on the GPU) is counted in the latency, which a real camera
would not add: pessimistic. If the worker is still busy when a tick is due, the tick is skipped —
the achieved rate is what the oracle judges. `min_latency_ms` holds results back further to
emulate a slower companion (this PC is not a Raspberry Pi 5).

**On the ground.** The camera looks down-forward 45° from ~0.2 m: it sees the ground at arm's
length and VPS cannot fix (measured: `no_features` below ~5 m). Until lift-off the companion
streams the package's take-off anchor (the surveyed spot and heading the vehicle is put down
in); from lift-off only VPS fixes are sent, the first one in the take-off column (the
localizability's take-off gap, flown on optical flow).

**The atlas** is loaded with the hash the package names; any other atlas (or a damaged file) and
the companion refuses: no anchor, no VPS — the autopilot cannot arm (`refusal`).
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from aerosplat.mission.models import VpsBinding
from aerosplat.sim.extnav import (LOSS_CONFIRM_S, MAV_ESTIMATOR_TYPE_VISION, MAV_FRAME_BODY_FRD,
                                  MAV_FRAME_LOCAL_FRD, Companion, Truth, ned_from_enu, quaternion)
from aerosplat.vps import perturb
from aerosplat.vps.atlas import Atlas, AtlasError
from aerosplat.vps.geometry import body_attitude, mount_axes, pose_error
from aerosplat.vps.localize import Localizer, PosePrior
from aerosplat.vps.route import CameraModel

ANCHOR_HZ = 10.0
PRIOR_SIGMA_CAP_M = 30.0
MAV_LANDED_STATE_ON_GROUND = 1


@dataclass
class _Job:
    k: int
    t: float                        # capture, simulation seconds
    truth: Truth
    omega_body: tuple[float, float, float]
    prior: PosePrior
    prior_err_m: float              # horizontal, prior camera centre vs truth (record only)
    level: str
    seed: int
    airborne: bool


class VpsCompanion(Companion):
    """`Companion` (origin, reactions, GNSS monitor) whose measurements come from the localizer.
    `world` renders the scene the camera sees (Renderer protocol of aerosplat.vps.build)."""

    def __init__(self, v: Any, *, binding: VpsBinding, atlas_path: Path, world: Any,
                 cam: CameraModel, readout_s: float, rate_hz: float, home_enu: tuple[float, float, float],
                 land_seq: Optional[int] = None, **kw: Any):
        super().__init__(v, **kw)
        loop = self.sim.vps
        assert loop is not None, "VpsCompanion needs nav.vps"
        self.loop, self.binding, self.world, self.cam = loop, binding, world, cam
        self.readout_s = readout_s if loop.rolling_shutter else 0.0
        self.period = 1.0 / rate_hz
        self.rate_hz = rate_hz
        self.loss_confirm = loop.loss_confirm_s or max(LOSS_CONFIRM_S, 2.5 * self.period)
        self.home_enu = np.asarray(home_enu, float)
        self.land_seq = land_seq
        self.refusal: Optional[str] = None
        self.localizer: Optional[Localizer] = None
        try:
            atlas = Atlas.load(atlas_path, expect_hash=binding.atlas.content_hash)
            intr = atlas.meta["intrinsics"]
            if (int(intr["width"]), int(intr["height"])) != (cam.width, cam.height) \
                    or abs(float(intr["fx"]) - cam.K[0]) > 1e-3 * cam.K[0]:
                raise AtlasError(f"atlas intrinsics {intr['width']}x{intr['height']} fx {intr['fx']:.1f} "
                                 f"are not the camera's {cam.width}x{cam.height} fx {cam.K[0]:.1f}")
            self.localizer = Localizer(atlas)
        except (AtlasError, OSError) as e:
            self.refusal = f"VPS atlas refused: {e}"
        # estimate (what the autopilot believes) and flight state, from MAVLink
        self.est_pos: Optional[Any] = None
        self.est_att: Optional[Any] = None
        self.landed_state: Optional[int] = None
        self.mission_seq: Optional[int] = None
        self.airborne_once = False
        # effects of faults (vps_degrade, vps_prior_offset)
        self.level_override: Optional[str] = None
        self.prior_offset: tuple[float, float] = (0.0, 0.0)
        # the worker
        self._jobs: "queue.Queue[Optional[_Job]]" = queue.Queue(maxsize=1)
        self._results: "queue.Queue[tuple[_Job, Any, float, float]]" = queue.Queue()
        self._busy = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._next_capture = -math.inf
        self._k = 0
        self._pending: list[tuple[_Job, Any, float, float]] = []
        self._last_anchor_wall = -math.inf
        self._last_fix_sent_t: Optional[float] = None
        self.records: list[dict[str, Any]] = []
        self.anchor_sent = 0
        self.anchor_err_m: list[float] = []
        self.skipped_busy = 0
        self.errors: list[str] = []

    # ── wiring ───────────────────────────────────────────────────────────
    def attach(self) -> None:
        super().attach()
        if self.refusal is not None:
            self.log(self.refusal)
            self.event(0.0, "nav", f"companion refuses, no ExternalNav: {self.refusal}")
            return
        self._worker = threading.Thread(target=self._work, name="vps-worker", daemon=True)
        self._worker.start()

    def detach(self) -> None:
        super().detach()
        if self._worker is not None:
            self._jobs.put(None)
            self._worker.join(timeout=10.0)
            self._worker = None

    def on(self, msg: Any) -> None:
        kind = msg.get_type()
        if kind == "GLOBAL_POSITION_INT":
            self.est_pos = msg
        elif kind == "ATTITUDE":
            self.est_att = msg
        elif kind == "EXTENDED_SYS_STATE":
            self.landed_state = int(msg.landed_state)
            if self.landed_state != MAV_LANDED_STATE_ON_GROUND and self.v.armed:
                self.airborne_once = True
        elif kind == "MISSION_CURRENT":
            self.mission_seq = int(msg.seq)
        super().on(msg)
        # Measurements go out whenever the link is pumped — also while the runner waits for
        # pre-arm and arming, before its flight loop calls tick() (as SIM-2n's do from SIM_STATE).
        if self.refusal is None:
            self._anchor()
            self._drain()

    # ── effects (faults) ─────────────────────────────────────────────────
    def set_effect(self, effect: dict) -> None:
        if "level" in effect:
            self.level_override = str(effect["level"])
        if "prior_offset_m" in effect:
            n, e = effect["prior_offset_m"]
            self.prior_offset = (float(n), float(e))

    def clear_effect(self, effect: dict) -> None:
        if "level" in effect:
            self.level_override = None
        if "prior_offset_m" in effect:
            self.prior_offset = (0.0, 0.0)

    # ── captures ─────────────────────────────────────────────────────────
    def _truth(self, msg: Any) -> None:
        """SIM_STATE: the camera's pose. Schedules a capture when one is due and the worker is
        free (a busy worker skips the tick: the rate drops, as it would on the companion)."""
        lat_i, lon_i = getattr(msg, "lat_int", 0), getattr(msg, "lon_int", 0)
        if lat_i == 0 and lon_i == 0:
            return
        try:
            t = self.now()
        except Exception:
            return
        from aerosplat.mission.export import enu_from_geodetic

        enu = enu_from_geodetic(self.frame, lat_i * 1e-7, lon_i * 1e-7, float(msg.alt))[0]
        tr = Truth(t=t, enu=np.asarray(enu, float), vel_ned=(float(msg.vn), float(msg.ve), float(msg.vd)),
                   roll=float(msg.roll), pitch=float(msg.pitch), yaw=float(msg.yaw))
        self.truth = tr
        if self.refusal is not None or not self.sim.extnav or t < self._next_capture:
            return
        # The schedule keeps its own grid (SIM_STATE comes at 10 Hz: re-anchoring on the sample
        # would turn 3 Hz into 2.5 Hz); a tick that cannot be kept is dropped, not caught up.
        nxt = self._next_capture + self.period
        self._next_capture = nxt if t < nxt else t + self.period
        if self._busy.is_set() or self._jobs.full():
            self.skipped_busy += 1
            return
        prior, prior_err = self._prior(t, tr)
        if prior is None:
            return
        self._k += 1
        level = self.level_override or self.loop.level
        job = _Job(k=self._k, t=t, truth=tr,
                   omega_body=(float(msg.xgyro), float(msg.ygyro), float(msg.zgyro)),
                   prior=prior, prior_err_m=prior_err, level=level,
                   seed=(self.sim.seed * 1_000_003 + self._k) % (2 ** 31 - 1),
                   airborne=self.airborne_once and self.landed_state != MAV_LANDED_STATE_ON_GROUND)
        self._busy.set()
        self._jobs.put(job)

    def _prior(self, t: float, tr: Truth) -> tuple[Optional[PosePrior], float]:
        """The camera pose the AUTOPILOT believes in (+ a fault's offset), and its σ: the
        configured σ after a fix, growing without one."""
        pos, att = self.est_pos, self.est_att
        if pos is None or att is None or (pos.lat == 0 and pos.lon == 0):
            return None, math.nan
        from aerosplat.mission.export import enu_from_geodetic

        e, n, _ = enu_from_geodetic(self.frame, pos.lat * 1e-7, pos.lon * 1e-7, 0.0)[0]
        up = float(self.home_enu[2]) + pos.relative_alt / 1000.0      # baro height above home
        dn, de = self.prior_offset
        C = (float(e) + de, float(n) + dn, up)
        ref = self._last_fix_sent_t
        since = t - ref if ref is not None else 0.0
        sigma = min(self.loop.prior_sigma_m + self.loop.prior_growth_mps * max(since, 0.0),
                    PRIOR_SIGMA_CAP_M)
        view = self.cam.view(C, float(att.roll), float(att.pitch), float(att.yaw))
        prior = PosePrior(C=view.C, yaw_deg=view.yaw_deg, pitch_deg=view.pitch_deg,
                          roll_deg=view.roll_deg, sigma_pos_m=sigma, sigma_yaw_deg=self.loop.prior_yaw_deg)
        err = float(math.hypot(C[0] - tr.enu[0], C[1] - tr.enu[1]))
        return prior, err

    def _work(self) -> None:
        """Worker thread: render, camera effects, localize. Nothing here touches MAVLink."""
        from aerosplat.vps.features import require_cv2

        cv2 = require_cv2()
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                t0 = time.perf_counter()
                tr = job.truth
                view = self.cam.view(tr.enu, tr.roll, tr.pitch, tr.yaw)
                render = getattr(self.world, "render_rgb", None)
                rgb = render(view, self.cam.width, self.cam.height, self.cam.K) if render else \
                    self.world.render(view, self.cam.width, self.cam.height, self.cam.K)[0]
                t1 = time.perf_counter()
                img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                w = mount_axes(self.cam.mount_pitch_deg,
                               self.cam.mount_yaw_deg) @ np.asarray(job.omega_body)
                omega_cam = (float(w[0]), float(w[1]), float(w[2]))
                img = perturb.rolling_shutter(img, omega_cam, self.readout_s, self.cam.K)
                img = perturb.apply(img, perturb.draw(job.level, np.random.default_rng(job.seed)))
                assert self.localizer is not None
                fix = self.localizer.localize(img, self.cam.K, job.prior)
                t2 = time.perf_counter()
                self._results.put((job, fix, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0))
            except Exception as e:          # a crash here is a VPS failure, not a silent hang
                self._results.put((job, e, 0.0, 0.0))
            finally:
                self._busy.clear()

    # ── the loop ─────────────────────────────────────────────────────────
    def tick(self) -> None:
        if self.refusal is None:
            self._anchor()
            self._drain()
        super().tick()          # the reactions (ADR-0015 §2.5)

    def on_ground(self) -> bool:
        return not self.airborne_once

    def _anchor(self) -> None:
        """Before lift-off: the package's take-off pose (ADR-0015 §8), at ANCHOR_HZ."""
        if self.airborne_once or not self.sim.extnav or self.v.target_system == 0:
            return
        wall = time.monotonic()
        if wall - self._last_anchor_wall < 1.0 / ANCHOR_HZ:
            return
        try:
            t = self.now()
        except Exception:
            return
        self._last_anchor_wall = wall
        a = self.binding.anchor
        att = self.est_att
        roll, pitch = (float(att.roll), float(att.pitch)) if att is not None else (0.0, 0.0)
        yaw = math.radians(a.heading_deg)
        n, e, d = ned_from_enu(a.enu)
        cov = [0.0] * 21
        cov[0] = cov[6] = a.sigma_m ** 2
        cov[11] = 1.0
        cov[15] = cov[18] = math.radians(2.0) ** 2
        cov[20] = math.radians(a.sigma_yaw_deg) ** 2
        nan = float("nan")
        # Stamped the declared latency back, like a VPS fix that took exactly that long: ArduPilot
        # maps companion time to its own by the SMALLEST lag it has seen (JitterCorrection) and
        # then subtracts VISO_DELAY_MS. A zero-lag anchor would set that baseline at 0, and every
        # fix after it would be fused VISO_DELAY_MS before its capture (SITL, window mission: EKF
        # vs truth p95 0.89 m before this, 0.75 m after). The pose does not move on the ground.
        stamp = t - self.latency
        self.v.conn.mav.odometry_send(int(stamp * 1e6), MAV_FRAME_LOCAL_FRD, MAV_FRAME_BODY_FRD,
                                      n, e, d, quaternion(roll, pitch, yaw), nan, nan, nan,
                                      nan, nan, nan, cov, [nan] * 21, 0, MAV_ESTIMATOR_TYPE_VISION, 100)
        self.anchor_sent += 1
        if self.truth is not None:
            self.anchor_err_m.append(float(math.hypot(a.enu[0] - self.truth.enu[0],
                                                      a.enu[1] - self.truth.enu[1])))
        self.last_capture = t
        self._fix_state(t, True)

    def _drain(self) -> None:
        while True:
            try:
                self._pending.append(self._results.get_nowait())
            except queue.Empty:
                break
        if not self._pending:
            return
        try:
            now = self.now()
        except Exception:
            return
        # Never earlier than the declared latency (VISO_DELAY_MS): ArduPilot's jitter correction
        # takes the smallest lag it sees as the transport's, so one early fix would shift every
        # later one (see _anchor). Results that took longer keep their capture time.
        hold = max(self.loop.min_latency_ms / 1000.0, self.latency)
        keep = []
        for item in self._pending:
            if now < item[0].t + hold:
                keep.append(item)
            else:
                self._result(now, *item)
        self._pending = keep

    def _result(self, now: float, job: _Job, fix: Any, render_ms: float, loc_ms: float) -> None:
        rec: dict[str, Any] = {"k": job.k, "t": round(job.t, 3), "latency_s": round(now - job.t, 3),
                               "render_ms": round(render_ms, 1), "localize_ms": round(loc_ms, 1),
                               "level": job.level, "prior_err_m": round(job.prior_err_m, 2),
                               "prior_sigma_m": round(job.prior.sigma_pos_m, 2),
                               "airborne": job.airborne, "sent": False}
        if isinstance(fix, Exception):
            rec.update(ok=False, reason="error", detail=repr(fix))
            self.errors.append(repr(fix))
        else:
            rec.update(ok=bool(fix.ok), reason=fix.reason)
            if fix.ok:
                view = self.cam.view(job.truth.enu, job.truth.roll, job.truth.pitch, job.truth.yaw)
                err = pose_error(fix.R, fix.C, view.R, np.asarray(view.C, float))
                rec.update(err_m=round(err["pos_m"], 3), err_h_m=round(err["horiz_m"], 3),
                           err_yaw_deg=round(err["yaw_deg"], 2), sigma_h_m=round(float(fix.sigma_h_m), 3),
                           inliers=fix.stats.get("inliers"))
            elif fix.candidate is not None:
                c = np.asarray(fix.candidate["C"], float)
                rec["candidate_err_h_m"] = round(float(np.hypot(*(c[:2] - job.truth.enu[:2]))), 2)
        self.records.append(rec)
        if job.t <= self.last_capture:
            return                      # older than what the anchor / a later result said
        if not self.airborne_once:
            return                      # on the ground the anchor speaks (VPS results recorded)
        ok = rec["ok"]
        if ok:
            self._send_fix(job, fix)
            rec["sent"] = True
        self.last_capture = job.t
        self._fix_state(job.t, ok)

    def _send_fix(self, job: _Job, fix: Any) -> None:
        C, cov = np.asarray(fix.C, float), np.asarray(fix.cov, float)
        roll, pitch, yaw = body_attitude(fix.R, self.cam.mount_pitch_deg, self.cam.mount_yaw_deg)
        n, e, d = ned_from_enu(C)
        # ENU (e, n, u) -> NED (n, e, d): swap the first two, flip the sign of everything with u
        P = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        pc = P @ cov[:3, :3] @ P.T
        pose_cov = [0.0] * 21
        pose_cov[0], pose_cov[1], pose_cov[2] = pc[0, 0], pc[0, 1], pc[0, 2]
        pose_cov[6], pose_cov[7] = pc[1, 1], pc[1, 2]
        pose_cov[11] = pc[2, 2]
        tilt = 0.5 * float(cov[3, 3] + cov[4, 4])
        pose_cov[15] = pose_cov[18] = tilt
        pose_cov[20] = math.radians(float(fix.sigma_yaw_deg)) ** 2
        nan = float("nan")
        self.v.conn.mav.odometry_send(int(job.t * 1e6), MAV_FRAME_LOCAL_FRD, MAV_FRAME_BODY_FRD,
                                      n, e, d, quaternion(roll, pitch, yaw), nan, nan, nan,
                                      nan, nan, nan, pose_cov, [nan] * 21, 0,
                                      MAV_ESTIMATOR_TYPE_VISION, 100)
        self.nav.measurements += 1
        tn, te, _ = ned_from_enu(job.truth.enu)
        self.nav.meas_err_h.append(math.hypot(n - tn, e - te))
        self.last_vps_ned = (n, e)
        self._last_fix_sent_t = job.t

    # ── landing on flow (Reaction.landing_on_flow) ──────────────────────────
    def landing(self) -> bool:
        if self.v.mode == "LAND":
            return True
        return (self.v.mode == "AUTO" and self.land_seq is not None
                and self.mission_seq == self.land_seq)

    # ── report ───────────────────────────────────────────────────────────
    def vps_summary(self, false_fix_m: float = 2.0) -> dict[str, Any]:
        air = [r for r in self.records if r["airborne"]]
        sent = [r for r in air if r["sent"]]
        false = [r for r in sent if r.get("err_m", 0.0) > false_fix_m]
        reasons: dict[str, int] = {}
        for r in air:
            if not r["ok"]:
                reasons[str(r["reason"])] = reasons.get(str(r["reason"]), 0) + 1

        def pct(vals: list[float], q: float) -> Optional[float]:
            return None if not vals else round(float(np.percentile(vals, q)), 3)

        span = (air[-1]["t"] - air[0]["t"]) if len(air) > 1 else 0.0
        lat = [r["latency_s"] for r in air]
        loc = [r["localize_ms"] for r in air]
        return {"refusal": self.refusal, "atlas_hash": self.binding.atlas.content_hash,
                "rate_hz_profile": self.rate_hz, "frames": len(self.records), "frames_airborne": len(air),
                "rate_hz": round((len(air) - 1) / span, 2) if span > 0 else None,
                "skipped_busy": self.skipped_busy, "sent": len(sent),
                "fix_rate": round(len(sent) / len(air), 3) if air else None,
                "refusals": dict(sorted(reasons.items())), "false_fixes": len(false),
                "false_fix_m": false_fix_m,
                "err_m_p50": pct([r["err_m"] for r in sent], 50),
                "err_m_p95": pct([r["err_m"] for r in sent], 95),
                "err_m_max": max((r["err_m"] for r in sent), default=None),
                "sigma_h_m_p50": pct([r["sigma_h_m"] for r in sent], 50),
                "latency_s_p50": pct(lat, 50), "latency_s_p95": pct(lat, 95),
                "localize_ms_p50": pct(loc, 50), "localize_ms_p95": pct(loc, 95),
                "render_ms_p50": pct([r["render_ms"] for r in air], 50),
                "prior_err_m_max": max((r["prior_err_m"] for r in air), default=None),
                "anchor_sent": self.anchor_sent,
                "anchor_err_m_max": max(self.anchor_err_m, default=None),
                "down_range_m_last": self.down_range_m,
                "loss_confirm_s": self.loss_confirm, "errors": self.errors[:5],
                "level": self.loop.level, "rolling_shutter_s": self.readout_s,
                "min_latency_ms": self.loop.min_latency_ms}

    def summary(self) -> dict:
        out = super().summary()
        out["vps"] = {**self.vps_summary(), "records": list(self.records)}   # report.json
        return out


def make_world(run_dir: Path) -> Any:
    """The site run's splat as the renderer of the camera's world (GPU, in-process)."""
    from aerosplat.vps.twin import EngineRenderer, SiteRun

    return EngineRenderer(SiteRun.open(Path(run_dir)))


Log = Callable[[str], Any]
