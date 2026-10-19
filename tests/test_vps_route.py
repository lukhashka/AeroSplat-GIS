"""SIM-3 without SITL or GPU (ADR-0015 §5 stage 2): the planned route, the corridor atlas and its
localizability on a synthetic textured plane, `vps_gap` findings in the plan, the camera on the
vehicle, the rolling shutter, the in-the-loop companion against a fake vehicle, the scenario
model, the runner's preflight for VPS packages and the `vps` oracle."""

import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

import mission_scenes as ms
import sim_scenarios as ss
from aerosplat.geo.crs import EnuFrame
from aerosplat.mission.export import ExportRefused, geodetic, nav_parameters, render
from aerosplat.mission.models import Code, ExtNavSpec, NavProfile, Verdict
from aerosplat.sim.oracles import FAIL, PASS, NavContext, check_vps
from aerosplat.sim.scenario import Expect, NavSim, Scenario
from aerosplat.vps import perturb
from aerosplat.vps.atlas import Atlas
from aerosplat.vps.features import make_extractor
from aerosplat.vps.geometry import angles, body_attitude, camera_rotation, rotation
from aerosplat.vps.route import (AtlasLayout, CameraModel, Gap, QueryResult, RoutePose, attach,
                                 build_route_atlas, find_gaps, localizability, prepare, route_poses)
from vps_scenes import PlaneRenderer

CAM = CameraModel(width=640, height=360, K=(460.0, 460.0, 319.5, 179.5), mount_pitch_deg=-45.0,
                  name="imx290:4mm")
LIGHT = AtlasLayout(spacing_m=3.0, yaws=(0, 45, -45, 90, -90, 180), near_spacing_m=1.5,
                    near_lateral_m=(0.0,), near_yaws=(0, 15, -15, 180), hover_along_m=(0.0,))
COMPLETE = {"outcome": "complete", "reason": "delivered"}


class LoweredPlane(PlaneRenderer):
    """The textured plane 25 m below the street's ground: the street mission flies 9 m above
    home, and the plane's texture (10 cm per texel) needs the bench's 25-60 m of distance."""

    DROP_M = 25.0

    def render(self, view, width, height, K):
        from aerosplat.vps.build import View

        up = View((view.C[0], view.C[1], view.C[2] + self.DROP_M), view.yaw_deg, view.pitch_deg,
                  view.roll_deg)
        return super().render(up, width, height, K)


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    """The street window mission, GNSS-free, VPS declared as a pose-only source at 3 Hz."""
    p = ss.nav_plan(tmp_path_factory.mktemp("plan"), gnss="none")
    ext = ExtNavSpec(rate_hz=3.0, latency_ms=100.0, velocity=False)
    prof = p.profile.model_copy(update={"nav": NavProfile(gnss="none", extnav=ext, flow=True)})
    return p.model_copy(update={"profile": prof})


@pytest.fixture(scope="module")
def plane():
    return LoweredPlane(seed=3)


@pytest.fixture(scope="module")
def built(plan, plane, tmp_path_factory):
    folder = tmp_path_factory.mktemp("pkg")
    out, atlas, rep = prepare(plan, plane, CAM, make_extractor("orb", n_max=600), folder,
                              run_id="synthetic", layout=LIGHT)
    return folder, out, atlas, rep


# ── the camera on the vehicle ────────────────────────────────────────────
def test_a_level_vehicle_looks_down_the_mount_and_the_attitude_comes_back():
    R = camera_rotation(0.0, 0.0, math.radians(30.0), -45.0)
    assert np.allclose(R, rotation(30.0, -45.0, 0.0), atol=1e-12)
    for r, p, y in ((0.1, -0.05, 2.0), (-0.2, 0.15, -1.0), (0.0, 0.0, 0.3)):
        back = body_attitude(camera_rotation(r, p, y, -45.0, 10.0), -45.0, 10.0)
        assert np.allclose(back, (r, p, y), atol=1e-9)
        # the render service's angles describe the same camera
        assert np.allclose(rotation(*angles(camera_rotation(r, p, y, -45.0))),
                           camera_rotation(r, p, y, -45.0), atol=1e-9)


def test_rolling_shutter_is_identity_at_rest_and_skews_rows_with_a_turn():
    img = np.tile(np.arange(200, dtype=np.uint8), (100, 1))[..., None].repeat(3, 2)
    K = (150.0, 150.0, 99.5, 49.5)
    assert perturb.rolling_shutter(img, (0.0, 0.0, 0.0), 0.03, K) is img
    out = perturb.rolling_shutter(img, (0.0, 1.0, 0.0), 0.03, K)     # pan right at 1 rad/s
    shift_top = float(out[0, 100, 0]) - float(img[0, 100, 0])
    shift_bottom = float(out[-1, 100, 0]) - float(img[-1, 100, 0])
    # fx·ω·readout/2 = 2.25 px either way of the middle row, opposite signs
    assert shift_top == pytest.approx(-shift_bottom, abs=1.0) and abs(shift_top) == pytest.approx(2.25, abs=1.0)


# ── the planned route ────────────────────────────────────────────────────
def test_the_route_is_the_planned_flight_with_its_headings_and_holds(plan):
    poses = route_poses(plan, spacing_m=2.0)
    home = np.asarray(plan.home.enu)
    assert np.allclose(poses[0].C, home) and np.allclose(poses[-1].C, home)
    assert [p.segment for p in poses][0] == "takeoff" and poses[-1].segment == "landing"
    segs = list(dict.fromkeys(p.segment for p in poses))
    assert segs.index("takeoff") < segs.index("approach") < segs.index("hover") < segs.index("landing")
    assert all(b.t >= a.t and b.s >= a.s for a, b in zip(poses, poses[1:]))
    assert all(p.yaw_deg == 0.0 for p in poses if p.segment == "takeoff")        # placed facing north
    hover = [p for p in poses if p.segment == "hover"]
    assert hover and all(p.yaw_deg == pytest.approx(plan.approach.heading_deg % 360) for p in hover)
    assert hover[-1].t - hover[0].t >= plan.profile.hold_s - 5.0 - 1e-6
    exit_ = [p for p in poses if p.segment == "exit"]
    assert exit_ and exit_[0].yaw_deg == pytest.approx((plan.approach.heading_deg + 180) % 360)
    # a time step samples the slow approach and the hold every second
    dense = route_poses(plan, spacing_m=4.0, max_dt_s=1.0, hold_step_s=1.0)
    assert max(b.t - a.t for a, b in zip(dense, dense[1:])) <= 1.0 + 1e-6
    assert max(b.s - a.s for a, b in zip(dense, dense[1:])) <= 4.0 + 1e-6


def _r(t, ok, seg="transit"):
    return QueryResult(s=t, t=t, segment=seg, height_m=10.0, C=(0.0, 0.0, 10.0), ok=ok, reason=None)


def _track(t_end=40.0, land_from=35.0):
    return [RoutePose(s=t, t=t, segment="landing" if t > land_from else ("takeoff" if t < 3 else "transit"),
                      C=(0.0, 0.0, 10.0), yaw_deg=0.0, height_m=10.0) for t in np.arange(0.0, t_end + 0.5, 1.0)]


def test_gaps_are_measured_fix_to_fix_and_block_beyond_the_drift_budget():
    track = _track()
    ok = {t: True for t in range(1, 40)}
    for t in (1, 2, 3):                 # take-off: first fix at 4 s -> 4 s on flow
        ok[t] = False
    for t in (10, 11):                  # 9 -> 12: 3 s
        ok[t] = False
    for t in range(20, 26):             # 19 -> 26: 7 s > 5 s
        ok[t] = False
    for t in range(36, 40):             # landing column from 35 s: reported
        ok[t] = False
    gaps = find_gaps([_r(float(t), v) for t, v in sorted(ok.items())], track, drift_budget_s=5.0)
    kinds = [(g.kind, round(g.duration_s, 1), g.blocking) for g in gaps]
    assert kinds == [("takeoff", 4.0, False), ("interior", 3.0, False), ("interior", 7.0, True),
                     ("landing", 5.0, False)]
    # a landing gap that begins in transit counts what is flown before the landing column
    ok2 = {t: t < 28 for t in range(1, 40)}
    g = find_gaps([_r(float(t), v) for t, v in sorted(ok2.items())], track, 5.0)[-1]
    assert g.kind == "landing" and g.counted_s == pytest.approx(8.0) and g.blocking
    # never a fix: one gap over everything, blocking
    [g] = find_gaps([_r(float(t), False) for t in range(1, 40)], track, 5.0)
    assert g.blocking and g.length_m == pytest.approx(40.0)


# ── atlas, localizability, the plan ──────────────────────────────────────
def test_the_corridor_atlas_localizes_the_route_and_is_bound_to_the_plan(built, plan):
    folder, out, atlas, rep = built
    assert atlas.meta["kind"] == "route" and atlas.meta["mission"] == plan.id
    assert rep.false_fixes == 0
    assert rep.segments["transit"]["fix_rate"] >= 0.8 and rep.segments["hover"]["fix_rate"] >= 0.8
    assert out.vps.atlas.content_hash == atlas.meta["content_hash"] == Atlas.load(folder / "vps" / "atlas.npz").content_hash()
    assert out.vps.anchor.enu == plan.home.enu and out.vps.anchor.heading_deg == 0.0
    assert out.vps.localizability["drift_budget_s"] == 5.0
    assert json.loads((folder / "vps" / "localizability.json").read_text(encoding="utf-8"))["rows"]
    assert not [f for f in out.findings if f.code is Code.vps_gap]
    assert out.verdict is plan.verdict


def test_localizability_is_deterministic(built, plane):
    _, out, atlas, rep = built
    track = route_poses(out, spacing_m=LIGHT.spacing_m, max_dt_s=1.0, hold_step_s=1.0)
    again = localizability(plane, atlas, track, CAM, drift_budget_s=5.0)
    assert [(q.ok, q.reason, q.err_m) for q in again.queries] == [(q.ok, q.reason, q.err_m) for q in rep.queries]
    assert again.summary() == rep.summary()


@pytest.mark.hazard("H-013")
def test_a_route_outside_the_atlas_is_a_blocking_vps_gap(built, plan, plane, tmp_path):
    """The same mission 150 m east: every frame looks at ground the atlas does not hold. The
    plan gets `vps_gap` (where, how long), becomes infeasible, and the export refuses it."""
    _, out, atlas, _ = built
    moved = [RoutePose(s=p.s, t=p.t, segment=p.segment, C=(p.C[0] + 150.0, p.C[1], p.C[2]),
                       yaw_deg=p.yaw_deg, height_m=p.height_m)
             for p in route_poses(out, spacing_m=4.0, max_dt_s=1.0, hold_step_s=1.0)]
    rep = localizability(plane, atlas, moved, CAM, drift_budget_s=5.0)
    assert sum(q.ok for q in rep.queries) == 0
    assert {q.reason for q in rep.queries} <= {"outside_atlas", "no_features"}
    bad = attach(plan, atlas, rep, CAM, run_id="synthetic")
    gaps = [f for f in bad.findings if f.code is Code.vps_gap]
    assert gaps and bad.verdict is Verdict.infeasible and gaps[0].at is not None
    assert "no VPS fix for" in gaps[0].detail and "drift budget" in gaps[0].detail
    with pytest.raises(ExportRefused, match="vps_gap"):
        render(bad, ms.ORIGIN, "parm", accept_risks=True)
    # re-attaching a good atlas replaces the old findings
    assert not [f for f in attach(bad, atlas, built[3], CAM, run_id="synthetic").findings
                if f.code is Code.vps_gap]


def test_attach_refuses_an_atlas_of_another_mission_or_a_gnss_plan(built, tmp_path):
    _, out, atlas, rep = built
    other = Atlas(**{k: getattr(atlas, k) for k in ("kf_R", "kf_C", "kf_look", "kf_offsets",
                                                    "kf_global", "xy", "desc", "xyz", "anchor")},
                  meta={**atlas.meta, "mission": "msn_00000000000000000000000000"})
    other.save(tmp_path / "o.npz")
    with pytest.raises(ValueError, match="built for mission"):
        attach(out, other, rep, CAM, run_id="synthetic")
    gnss = out.model_copy(update={"profile": out.profile.model_copy(update={"nav": NavProfile()})})
    with pytest.raises(ValueError, match="GNSS-free"):
        attach(gnss, atlas, rep, CAM, run_id="synthetic")


def test_the_package_names_its_atlas_and_takes_no_velocity_from_vps(built):
    _, out, atlas, _ = built
    _, _, plan_text = render(out, ms.ORIGIN, "plan", accept_risks=True)
    block = json.loads(plan_text)["aerosplat"]["vps"]
    assert block["atlas"]["content_hash"] == atlas.meta["content_hash"]
    assert block["atlas"]["file"] == "vps/atlas.npz" and block["anchor"]["heading_deg"] == 0.0
    params = dict((n, v) for n, v, _ in nav_parameters(out.profile.nav))
    assert params["EK3_SRC1_VELXY"] == 0 and params["EK3_SRC1_POSXY"] == 6 and params["VISO_DELAY_MS"] == 100


# ── the scenario model ───────────────────────────────────────────────────
def test_vps_in_the_loop_has_no_error_model_and_flies_at_speedup_1():
    with pytest.raises(ValueError, match="no error model"):
        NavSim(gnss="none", sigma_m=0.3, vps={"run_dir": "r"})
    with pytest.raises(ValueError, match="no error model"):
        NavSim(gnss="none", zones=[{"center": [0, 0], "radius_m": 5}], vps={"run_dir": "r"})
    with pytest.raises(ValueError, match="speedup 1"):
        ss.scenario("x", COMPLETE, nav={"gnss": "none", "vps": {"run_dir": "r"}}, speedup=3)
    with pytest.raises(ValueError, match="need VPS in the loop"):
        ss.scenario("x", COMPLETE, nav={"gnss": "none"},
                    faults=[{"fault": "vps_degrade", "at_s": 1, "level": "severe"}])
    with pytest.raises(ValueError, match="vps_degrade needs level"):
        ss.scenario("x", COMPLETE, nav={"gnss": "none", "vps": {"run_dir": "r"}},
                    faults=[{"fault": "vps_degrade", "at_s": 1, "level": "low"}])
    sc = ss.scenario("x", {"outcome": "abort", "reason": "vps_loss", "modes": ["GUIDED"]},
                     nav={"gnss": "none", "vps": {"run_dir": "r"}}, speedup=1,
                     faults=[{"fault": "vps_degrade", "at_s": 1, "level": "severe"},
                             {"fault": "vps_prior_offset", "at_s": 1, "offset_m": [15, 0]}])
    assert sc.nav.landing_on_flow and not NavSim(gnss="none").landing_on_flow
    assert NavSim(gnss="none", reaction={"landing_on_flow": True}).landing_on_flow
    assert Expect(outcome="refused", reason="atlas_mismatch").reason == "atlas_mismatch"
    assert isinstance(sc, Scenario)


# ── the in-the-loop companion against a fake vehicle ─────────────────────
class FakeMav:
    def __init__(self):
        self.sent = []

    def __getattr__(self, name):
        if name.endswith("_send"):
            return lambda *a: self.sent.append((name[:-5], a))
        raise AttributeError(name)


class FakeVehicle:
    target_system, target_component = 1, 1

    def __init__(self):
        self.conn = SimpleNamespace(mav=FakeMav())
        self.listeners = []
        self.t = 0.0
        self.armed = True
        self.mode = "AUTO"
        self.commands = []

    def sim_now(self, speedup):
        return self.t

    def command(self, cmd, *p):
        self.commands.append((cmd, p))
        return 0

    def set_mode(self, mode):
        self.mode = mode

    def sent(self, kind):
        return [a for k, a in self.conn.mav.sent if k == kind]


def _msg(kind, **kw):
    return SimpleNamespace(get_type=lambda: kind, **kw)


def _sim_state(enu, yaw_deg):
    lat, lon, h = geodetic(EnuFrame(ms.ORIGIN), [enu])
    return _msg("SIM_STATE", lat_int=int(round(float(lat[0]) * 1e7)), lon_int=int(round(float(lon[0]) * 1e7)),
                alt=float(h[0]), vn=0.0, ve=0.0, vd=0.0, roll=0.0, pitch=0.0, yaw=math.radians(yaw_deg),
                xgyro=0.0, ygyro=0.0, zgyro=0.0)


def _estimate(enu, yaw_deg, home_z):
    lat, lon, _ = geodetic(EnuFrame(ms.ORIGIN), [enu])
    return [_msg("GLOBAL_POSITION_INT", lat=int(round(float(lat[0]) * 1e7)), lon=int(round(float(lon[0]) * 1e7)),
                 relative_alt=int(round((enu[2] - home_z) * 1000))),
            _msg("ATTITUDE", roll=0.0, pitch=0.0, yaw=math.radians(yaw_deg))]


def vps_companion(built, plane, atlas_path=None, **nav):
    from aerosplat.sim.record import Recording
    from aerosplat.sim.vpsloop import VpsCompanion

    folder, out, _, _ = built
    v = FakeVehicle()
    sim = NavSim(gnss="none", vps={"run_dir": "unused", "level": "clean", **nav})
    c = VpsCompanion(v, binding=out.vps, atlas_path=atlas_path or folder / "vps" / "atlas.npz",
                     world=plane, cam=CAM, readout_s=0.0, rate_hz=3.0, home_enu=tuple(out.home.enu),
                     land_seq=None, origin=ms.ORIGIN, sim=sim, spec=out.profile.nav.extnav, flow=True,
                     grid=ms.street_grid(), speedup=1.0, event=Recording().event)
    c.attach()
    return v, c


def _fly(v, c, truth, est, yaw, seconds, landed_state=2, hz=10.0):
    home_z = float(c.home_enu[2])
    for _ in range(int(seconds * hz)):
        v.t += 1.0 / hz
        for m in _estimate(est, yaw, home_z) + [_msg("EXTENDED_SYS_STATE", landed_state=landed_state),
                                                _sim_state(truth, yaw)]:
            c.on(m)
        c.tick()
        time.sleep(1.0 / hz)        # the worker renders and localizes in wall-clock time
    deadline = time.monotonic() + 10.0
    while c._busy.is_set() and time.monotonic() < deadline:
        time.sleep(0.05)
    c.on(_msg("HEARTBEAT"))


@pytest.mark.hazard("H-013")
def test_the_anchor_until_lift_off_then_only_vps_fixes_at_capture_time(built, plane):
    folder, out, _, _ = built
    v, c = vps_companion(built, plane)
    try:
        home = tuple(out.home.enu)
        _fly(v, c, home, home, 0.0, 1.0, landed_state=1)               # on the ground
        anchors = v.sent("odometry")
        assert len(anchors) >= 5 and c.anchor_sent == len(anchors)
        assert math.hypot(anchors[-1][3] - home[1], anchors[-1][4] - home[0]) < 1e-6   # NED of the anchor
        # stamped the declared latency back (ArduPilot's jitter correction keys on the least lag)
        assert anchors[-1][0] / 1e6 <= v.t - out.profile.nav.extnav.latency_ms / 1000.0 + 1e-6
        hp = (0.0, 6.0, 9.0)                                            # in the air, over the plane
        n0 = len(anchors)
        _fly(v, c, hp, hp, 0.0, 3.0)
        assert c.anchor_sent == n0                                      # no anchor after lift-off
        fixes = v.sent("odometry")[n0:]
        recs = [r for r in c.records if r["airborne"]]
        assert recs and len(fixes) == sum(r["sent"] for r in recs) >= 3
        t_us, frame, child, x, y, z, q, vx = fixes[-1][:8]
        assert t_us / 1e6 in [r["t"] for r in recs if r["sent"]]        # the capture's time
        assert min(r["latency_s"] for r in recs) >= 0.1 - 1e-9          # never before VISO_DELAY_MS
        assert math.isnan(vx)                                           # a pose only
        assert math.hypot(x - hp[1], y - hp[0]) < 0.5
        cov = fixes[-1][13]
        assert 0 < cov[0] < 1.0 and cov[20] > 0
        assert all(r["err_m"] < 2.0 for r in recs if r["sent"])
        s = c.vps_summary()
        assert s["false_fixes"] == 0 and s["refusal"] is None
    finally:
        c.detach()


@pytest.mark.hazard("H-013")
def test_a_prior_15_m_off_never_feeds_a_false_fix(built, plane):
    """The prior is the autopilot's estimate, not the truth: an estimate 15 m off makes the
    localizer refuse (prior_mismatch / no candidates) or find the true pose; it never sends a
    pose near the wrong prior."""
    _, out, _, _ = built
    v, c = vps_companion(built, plane)
    try:
        _fly(v, c, out.home.enu, out.home.enu, 0.0, 0.3, landed_state=1)
        truth, wrong = (0.0, 6.0, 9.0), (0.0, 21.0, 9.0)
        _fly(v, c, truth, wrong, 0.0, 3.0)
        recs = [r for r in c.records if r["airborne"]]
        assert recs and all(r["prior_err_m"] == pytest.approx(15.0, abs=0.1) for r in recs)
        assert all(r["err_m"] < 2.0 for r in recs if r["sent"])
        assert all(r["reason"] in ("prior_mismatch", "outside_atlas", "too_few_matches",
                                   "too_few_inliers", "low_inlier_ratio") for r in recs if not r["ok"])
    finally:
        c.detach()


@pytest.mark.hazard("H-013")
def test_another_atlas_than_the_package_names_and_the_companion_refuses_to_stream(built, plane, tmp_path):
    folder, out, atlas, _ = built
    other = Atlas.load(folder / "vps" / "atlas.npz")
    other.meta = {**other.meta, "note": "another build"}
    other.save(tmp_path / "other.npz")
    v, c = vps_companion(built, plane, atlas_path=tmp_path / "other.npz")
    try:
        assert c.refusal and "is not the expected" in c.refusal
        _fly(v, c, out.home.enu, out.home.enu, 0.0, 1.0, landed_state=1)
        assert v.sent("odometry") == [] and c.anchor_sent == 0
        assert c.vps_summary()["refusal"] == c.refusal
    finally:
        c.detach()


# ── the oracle ───────────────────────────────────────────────────────────
def _ctx(records, min_rate=2.0, **tol):
    """What check_vps reads of a Context: the navigation context and the tolerances."""
    return SimpleNamespace(nav=NavContext(gnss="none", flow=True, fix=[], gnss_fixes=0,
                                          vps_records=records, vps_min_rate_hz=min_rate),
                           tol=Expect(outcome="complete", reason="delivered", tolerances=tol).tolerances)


def _rec(t, sent=True, err=0.1):
    return {"t": t, "airborne": True, "ok": sent, "sent": sent, "err_m": err if sent else None}


def test_the_vps_oracle_fails_a_false_fix_and_a_low_rate():
    good = [_rec(i / 3.0) for i in range(60)]
    v = check_vps(None, _ctx(good))
    assert v.status == PASS and v.metrics["rate_hz"] == pytest.approx(3.0) and v.metrics["false_fixes"] == 0
    bad = good[:30] + [_rec(10.0, err=2.6)] + good[31:]
    v = check_vps(None, _ctx(bad))
    assert v.status == FAIL and "false fix" in v.detail and v.metrics["false_fix_t"] == [10.0]
    slow = [_rec(i / 1.5) for i in range(30)]
    v = check_vps(None, _ctx(slow))
    assert v.status == FAIL and "below the profile" in v.detail
    refused = [_rec(i / 3.0, sent=False) for i in range(60)]           # refusals are no fault
    assert check_vps(None, _ctx(refused)).status == PASS


def test_runner_preflight_needs_the_atlas_in_a_pose_only_package(built, tmp_path):
    import shutil

    from aerosplat.sim.runner import MissionPackage, Refusal, Runner

    folder, out, _, _ = built
    pkg = ss.export(out, tmp_path / "m")
    shutil.copytree(folder / "vps", pkg / "vps")
    sc = ss.scenario("t", COMPLETE, vehicle="x500-pi5-imx290", speedup=1,
                     nav={"gnss": "none", "vps": {"run_dir": str(tmp_path)}})
    with pytest.raises(Exception, match="speedup 1"):
        Runner(sc, MissionPackage.load(pkg), ms.street_grid(), provider=None, speedup=3)
    r = Runner(sc, MissionPackage.load(pkg), ms.street_grid(), provider=None, world=object())
    # the synthetic atlas was rendered for 480x270, the test bed's camera is 960x540
    with pytest.raises(Refusal, match="rendered for") as e:
        r.preflight()
    assert e.value.reason == "mission_mismatch"
    (pkg / "vps" / "atlas.npz").unlink()
    with pytest.raises(Refusal) as e:
        Runner(sc, MissionPackage.load(pkg), ms.street_grid(), provider=None, world=object()).preflight()
    assert e.value.reason == "package_incomplete"
    plain = ss.export(ss.nav_plan(tmp_path / "p", gnss="none"), tmp_path / "plain")
    with pytest.raises(Refusal) as e:
        Runner(sc, MissionPackage.load(plain), ms.street_grid(), provider=None, world=object()).preflight()
    assert e.value.reason == "mission_mismatch" and "names no VPS atlas" in e.value.detail


def test_build_route_atlas_skips_the_ground_and_records_its_layout(plan, plane):
    atlas, st = build_route_atlas(plane, plan, CAM, make_extractor("orb", n_max=300), {"run": "x"},
                                  AtlasLayout(spacing_m=6.0, yaws=(0,), near_segments=(), min_height_m=2.0))
    assert atlas.meta["layout"]["spacing_m"] == 6.0
    assert float(atlas.kf_C[:, 2].min()) >= plan.home.enu[2] + 2.0 - 1e-6
    assert st["keyframes"] == atlas.n_keyframes


def test_gap_describes_where_and_how_long():
    g = Gap("interior", 10.0, 22.0, 5.0, 12.5, ["transit"], (1.0, 2.0, 3.0), 7.5, True)
    assert g.length_m == 12.0 and g.duration_s == 7.5 and "7.5 s / 12.0 m" in g.describe()
