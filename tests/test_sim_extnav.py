"""L0 tests of SIM-2n (ADR-0015 stage 1) without SITL: the navigation profile in the plan and the
package, the runner's refusals for GNSS-free packages, the scenario model, the companion
emulator against a fake vehicle (measurements, origin, VPS-loss reactions with the rangefinder
stand-in), and the two new oracles."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import mission_scenes as ms
import sim_scenarios as ss
from aerosplat.geo.crs import EnuFrame
from aerosplat.mission.export import (BOOT_PARAMS, ExportRefused, geodetic, nav_parameters,
                                      nav_problems, parameters, read_parm, render)
from aerosplat.mission.models import ExtNavSpec, GnssUse, NavProfile, PlanningProfile
from aerosplat.sim.extnav import (CMD_SET_EKF_SOURCE_SET, Companion, GaussMarkov,
                                  MAV_FRAME_BODY_FRD, MAV_FRAME_LOCAL_FRD, ned_from_enu,
                                  quaternion, rotation, wall_from_ranges)
from aerosplat.sim.faults import injection, response
from aerosplat.sim.oracles import (NavContext, PASS, FAIL, check_navigation, check_wall_descent,
                                   judge)
from aerosplat.sim.params import load_snapshot
from aerosplat.sim.runner import MissionPackage, Refusal, Runner
from aerosplat.sim.scenario import Fault, NavSim, Reaction, Scenario

COMPLETE = {"outcome": "complete", "reason": "delivered"}
VPS_ABORT = {"outcome": "abort", "reason": "vps_loss", "modes": ["GUIDED"]}


@pytest.fixture(scope="module")
def grid():
    return ms.street_grid()


@pytest.fixture(scope="module")
def gnss_plan(tmp_path_factory):
    return ss.window_plan(tmp_path_factory.mktemp("plan"))


@pytest.fixture(scope="module")
def nav_plan(tmp_path_factory):
    return ss.nav_plan(tmp_path_factory.mktemp("plan-nav"), gnss="none")


@pytest.fixture
def nav_dir(tmp_path, nav_plan):
    return ss.export(nav_plan, tmp_path / "missions" / nav_plan.id)


def nav_runner(mdir, grid, expect=COMPLETE, **kw) -> Runner:
    kw.setdefault("vehicle", "sitl-quad-extnav")
    kw.setdefault("nav", {"gnss": "none"})
    return Runner(ss.scenario("t", expect, **kw), MissionPackage.load(mdir), grid, provider=None)


# ── the profile, the plan and the package ────────────────────────────────
def test_a_gnss_free_profile_needs_extnav_and_a_gnss_profile_takes_none():
    with pytest.raises(ValueError, match="needs an extnav"):
        NavProfile(gnss="none")
    with pytest.raises(ValueError, match="takes no extnav"):
        NavProfile(gnss="primary", extnav=ExtNavSpec())
    assert not NavProfile().gnss_free and NavProfile(gnss="monitor", extnav=ExtNavSpec()).gnss_free


def test_the_corridor_takes_the_navigation_sigma(gnss_plan, nav_plan):
    assert gnss_plan.sigma.nav_source == "gnss" and gnss_plan.sigma.gnss_m == 1.5
    assert nav_plan.sigma.nav_source == "extnav" and nav_plan.sigma.gnss_m == 0.5
    # σ_transit = hypot(σ_georef 0.3, σ_nav): a smaller radius with VPS than with GNSS.
    tr = {leg.name: leg.radius_m for leg in nav_plan.legs}
    tg = {leg.name: leg.radius_m for leg in gnss_plan.legs}
    assert tr["transit"] < tg["transit"]
    assert PlanningProfile().sigma_nav_m == 1.5


def test_a_gnss_package_is_unchanged_by_the_navigation_section(gnss_plan):
    names = {n for n, _, _ in parameters(gnss_plan)}
    assert not names & {"GPS1_TYPE", "VISO_TYPE", "ARMING_NEED_LOC"}
    assert not [n for n in names if n.startswith("EK3_")]
    assert nav_parameters(gnss_plan.profile.nav) == []


@pytest.mark.hazard("H-003", "H-013")
def test_the_gnss_free_package_is_valid_for_the_pinned_firmware(nav_plan):
    parm = {n: float(v) for n, v, _ in parameters(nav_plan)}
    fw = load_snapshot(variant="extnav")
    assert fw.problems(parm) == []
    assert parm["GPS1_TYPE"] == 0 and parm["VISO_TYPE"] == 1 and parm["ARMING_NEED_LOC"] == 1
    assert (parm["EK3_SRC1_POSXY"], parm["EK3_SRC1_VELXY"], parm["EK3_SRC1_YAW"]) == (6, 6, 6)
    assert (parm["EK3_SRC2_POSXY"], parm["EK3_SRC2_VELXY"], parm["EK3_SRC2_YAW"]) == (0, 5, 1)
    assert parm["EK3_SRC1_POSZ"] == parm["EK3_SRC2_POSZ"] == 1                   # barometer
    assert parm["VISO_DELAY_MS"] == 50 and parm["VISO_POS_M_NSE"] == 0.5
    # The base snapshot (VISO off) does not even know VISO_DELAY_MS: why the variant exists.
    assert "VISO_DELAY_MS" in load_snapshot().unknown(parm)
    assert BOOT_PARAMS <= set(load_snapshot(variant="extnav").params)
    assert nav_problems(nav_plan.profile.nav, parm) == []


@pytest.mark.hazard("H-013")
def test_a_package_that_would_fly_on_gnss_is_not_a_gnss_free_package(nav_plan):
    nav = nav_plan.profile.nav
    parm = {n: float(v) for n, v, _ in parameters(nav_plan)}
    assert nav_problems(nav, {**parm, "EK3_SRC1_POSXY": 3})            # GPS position
    assert nav_problems(nav, {**parm, "EK3_SRC3_YAW": 8})              # GSF needs GNSS
    assert nav_problems(nav, {k: v for k, v in parm.items() if k != "GPS1_TYPE"})
    monitor = NavProfile(gnss="monitor", extnav=ExtNavSpec(), flow=True)
    assert dict((n, v) for n, v, _ in nav_parameters(monitor))["GPS1_TYPE"] == 1


def test_the_export_refuses_a_package_that_misses_its_navigation(nav_plan, monkeypatch):
    import aerosplat.mission.export as ex

    _, _, text = render(nav_plan, ms.ORIGIN, "parm")
    assert read_parm(text)["EK3_SRC1_POSXY"] == 6 and "gnss none" in text
    real = ex.format_parm
    # A formatter that loses a line (the bug this gate is for): the export is refused.
    monkeypatch.setattr(ex, "format_parm", lambda plan: "\n".join(
        ln for ln in real(plan).splitlines() if not ln.startswith("EK3_SRC1_YAW")) + "\n")
    with pytest.raises(ExportRefused, match="EK3_SRC1_YAW missing"):
        render(nav_plan, ms.ORIGIN, "parm")


# ── the runner, offline ──────────────────────────────────────────────────
@pytest.mark.hazard("H-013")
def test_a_gnss_package_flown_as_gnss_free_is_refused(tmp_path, gnss_plan, grid):
    mdir = ss.export(gnss_plan, tmp_path / "m")
    r = nav_runner(mdir, grid)
    with pytest.raises(Refusal) as e:
        r.preflight()
    assert e.value.reason == "mission_mismatch" and "gnss none" in e.value.detail


@pytest.mark.hazard("H-013")
def test_a_tampered_gnss_free_package_is_refused(nav_dir, grid):
    text = (nav_dir / "mission.parm").read_text(encoding="utf-8")
    (nav_dir / "mission.parm").write_text(text.replace("EK3_SRC1_POSXY 6", "EK3_SRC1_POSXY 3"),
                                          encoding="utf-8")
    with pytest.raises(Refusal) as e:
        nav_runner(nav_dir, grid).preflight()
    assert e.value.reason == "mission_mismatch" and "EK3_SRC1_POSXY" in e.value.detail


def test_flow_fallback_needs_a_vehicle_with_flow(nav_dir, grid):
    with pytest.raises(Refusal) as e:
        nav_runner(nav_dir, grid, vehicle="sitl-quad").preflight()
    assert e.value.reason == "nav_config" and "FLOW_TYPE" in e.value.detail


def test_sitl_boots_with_the_navigation_and_the_world_without_gnss(nav_dir, grid):
    r = nav_runner(nav_dir, grid)
    pre = r.preflight()
    boot = r.boot_params()
    assert pre["nav"]["snapshot"] == "extnav" and pre["nav"]["boot"] == boot
    assert boot["GPS1_TYPE"] == 0 and boot["SIM_GPS1_ENABLE"] == 0 and boot["VISO_TYPE"] == 1
    assert boot["EK3_SRC1_POSXY"] == 6 and boot["EK3_SRC1_YAW"] == 6          # EKF3 starts on them
    assert boot["FLOW_TYPE"] == 10 and boot["RNGFND1_TYPE"] == 100 and boot["SIM_TERRAIN"] == 0
    assert r.navsim is not None and r.navsim.gnss is GnssUse.none


def test_a_gnss_run_boots_as_before(tmp_path, gnss_plan, grid):
    mdir = ss.export(gnss_plan, tmp_path / "m")
    r = Runner(ss.scenario("t", COMPLETE), MissionPackage.load(mdir), grid, provider=None)
    assert r.boot_params() == {} and r.navsim is None and not r.nav_run
    assert "nav" not in r.preflight()


def test_vps_loss_is_answered_by_the_companion(nav_dir, grid):
    r = nav_runner(nav_dir, grid, VPS_ABORT,
                   faults=[{"fault": "vps_loss", "after": "hover", "at_s": 12}],
                   nav={"gnss": "none", "reaction": {"drift_budget_s": 4}})
    pre = r.preflight()
    assert pre["response"]["modes"] == ["GUIDED"] and pre["response"]["delay_s"] == 4
    assert pre["faults"] == ["companion VPS gives no fix (ODOMETRY stops)"]


# ── scenario and faults ──────────────────────────────────────────────────
def test_scenario_rules_for_gnss_free_runs():
    base = {"scenario": "x", "expect": COMPLETE}
    with pytest.raises(ValueError, match="GNSS-free run"):
        Scenario.model_validate({**base, "faults": [{"fault": "vps_loss", "at_s": 1}]})
    with pytest.raises(ValueError, match="monitor"):
        Scenario.model_validate({**base, "nav": {"gnss": "none"},
                                 "faults": [{"fault": "gps_spoof", "at_s": 1, "offset_m": [20, 0]}]})
    with pytest.raises(ValueError, match="without GNSS"):
        Scenario.model_validate({**base, "nav": {"gnss": "none"},
                                 "faults": [{"fault": "gps_loss", "at_s": 1}]})
    with pytest.raises(ValueError, match="GNSS-free run"):
        Scenario.model_validate({**base, "nav": {"gnss": "primary"}})
    with pytest.raises(ValueError, match="offset_m"):
        Fault(fault="gps_spoof", at_s=1)
    # A no-fix zone can be the cause of a vps_loss abort, without any fault.
    Scenario.model_validate({"scenario": "x", "expect": VPS_ABORT,
                             "nav": {"gnss": "none", "zones": [{"center": [0, 10], "radius_m": 3}]}})


def test_gps_spoof_offsets_become_degrees_at_the_site():
    fw = load_snapshot(variant="extnav")
    f = Fault(fault="gps_spoof", at_s=1, offset_m=(20.0, 10.0))
    inj = injection(f, fw, {"GPS1_TYPE": 1}, lat_deg=50.45)
    d = {c.name: c.value for c in inj.inject}
    assert d["SIM_GPS1_GLTCH_X"] * 111195 == pytest.approx(20.0, rel=1e-3)
    assert d["SIM_GPS1_GLTCH_Y"] * 111195 * math.cos(math.radians(50.45)) == pytest.approx(10.0, rel=1e-3)
    assert {c.name: c.value for c in inj.clear} == {"SIM_GPS1_GLTCH_X": 0, "SIM_GPS1_GLTCH_Y": 0}
    with pytest.raises(Exception, match="no GNSS receiver"):
        injection(f, fw, {"GPS1_TYPE": 0}, lat_deg=50.45)
    with pytest.raises(Exception, match="no ExternalNav"):
        injection(Fault(fault="vps_loss", at_s=1), fw, {"VISO_TYPE": 0})


def test_vps_loss_response_without_flow_stops_at_once():
    resp = response("vps_loss", {}, {"EK3_SRC2_VELXY": 0}, reaction=Reaction(drift_budget_s=9))
    assert resp.modes == ("GUIDED",) and resp.delay_s == 0
    with pytest.raises(Exception, match="companion"):
        response("vps_loss", {}, {})


# ── pure pieces of the companion ─────────────────────────────────────────
def test_quaternion_and_rotation_agree():
    for r, p, y in ((0.1, -0.2, 1.3), (0.0, 0.0, -2.9), (-0.3, 0.25, 0.0)):
        w, x, yq, z = quaternion(r, p, y)
        R = np.array([[1 - 2 * (yq * yq + z * z), 2 * (x * yq - w * z), 2 * (x * z + w * yq)],
                      [2 * (x * yq + w * z), 1 - 2 * (x * x + z * z), 2 * (yq * z - w * x)],
                      [2 * (x * z - w * yq), 2 * (yq * z + w * x), 1 - 2 * (x * x + yq * yq)]])
        assert np.allclose(R, rotation(r, p, y), atol=1e-12)
    assert ned_from_enu((1.0, 2.0, 3.0)) == (2.0, 1.0, -3.0)


def test_wall_from_forward_ranges():
    d, n = wall_from_ranges((-30, 0, 30), (3 / math.cos(math.radians(30)), 3.0,
                                           3 / math.cos(math.radians(30))))
    assert d == pytest.approx(3.0) and n == pytest.approx((-1.0, 0.0), abs=1e-9)
    # A wall at 45° to the right-front: the normal points back-left.
    d, n = wall_from_ranges((0, 30), (4.0, 4.0 * math.sin(math.radians(45)) / math.sin(math.radians(105))))
    assert n[0] < 0 and n[1] < 0 and d == pytest.approx(4.0 * math.sin(math.radians(45)), rel=1e-6)
    assert wall_from_ranges((0,), (5.0,)) == (5.0, (-1.0, -0.0))
    assert wall_from_ranges((-30, 0, 30), (math.inf,) * 3) is None


def test_gauss_markov_is_stationary_with_its_sigma():
    import random

    gm = GaussMarkov(0.3, 5.0, random.Random(1), dims=1)
    xs = [gm.step(0.1)[0] for _ in range(60000)]
    assert np.std(xs) == pytest.approx(0.3, rel=0.1)
    lag = int(5.0 / 0.1)
    a = np.asarray(xs)
    assert np.corrcoef(a[:-lag], a[lag:])[0, 1] == pytest.approx(math.exp(-1), abs=0.1)


# ── the companion against a fake vehicle ─────────────────────────────────
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


def sim_state(enu, yaw_deg=0.0, vel=(0.0, 0.0, 0.0)):
    lat, lon, h = geodetic(EnuFrame(ms.ORIGIN), [enu])
    return SimpleNamespace(get_type=lambda: "SIM_STATE", lat_int=int(round(float(lat[0]) * 1e7)),
                           lon_int=int(round(float(lon[0]) * 1e7)), alt=float(h[0]),
                           vn=vel[0], ve=vel[1], vd=vel[2], roll=0.0, pitch=0.0,
                           yaw=math.radians(yaw_deg))


def companion(grid, **sim):
    v = FakeVehicle()
    events = []
    c = Companion(v, origin=ms.ORIGIN, sim=NavSim(gnss="none", **sim), spec=ExtNavSpec(), flow=True,
                  grid=grid, speedup=1.0, event=lambda *a, **k: events.append(a))
    c.attach()
    return v, c, events


def feed(v, c, enu, seconds, yaw_deg=0.0, hz=20.0, tick=True):
    for _ in range(int(seconds * hz)):
        v.t += 1.0 / hz
        c.on(sim_state(enu, yaw_deg))
        if tick:
            c.tick()


def test_measurements_are_odometry_from_the_truth_with_latency_and_honest_covariance(grid):
    v, c, _ = companion(grid, sigma_m=0.01, sigma_z_m=0.01, yaw_sigma_deg=0.01, vel_sigma_mps=0.01,
                        bias_sigma_m=0.2)
    feed(v, c, (3.0, 4.0, 9.0), 2.0, yaw_deg=90.0, tick=False)
    odo = v.sent("odometry")
    assert len(odo) == pytest.approx(19, abs=1)                     # 10 Hz for ~1.95 s
    t_us, frame, child, x, y, z, q = odo[-1][:7]
    assert (frame, child) == (MAV_FRAME_LOCAL_FRD, MAV_FRAME_BODY_FRD)
    assert t_us / 1e6 == pytest.approx(v.t - 0.05, abs=0.06)          # captured 50 ms earlier
    assert math.hypot(x - 4.0, y - 3.0) < 1.0 and z == pytest.approx(-9.0, abs=0.05)   # NED
    pose_cov = odo[-1][13]
    assert pose_cov[0] == pytest.approx(0.01 ** 2 + 0.2 ** 2)         # white + bias, honestly
    assert c.summary()["measurements"] == len(odo)


def test_the_origin_is_sent_until_it_reads_back(grid):
    v, c, events = companion(grid)
    hb = SimpleNamespace(get_type=lambda: "HEARTBEAT")
    c.on(hb)
    lat, lon, alt = c.origin_target()
    assert v.sent("set_gps_global_origin") == [(1, lat, lon, alt, 0)]
    c.on(SimpleNamespace(get_type=lambda: "GPS_GLOBAL_ORIGIN", latitude=lat, longitude=lon + 3,
                         altitude=alt))
    assert not c.origin_ok and "is not the Site origin" in c.origin_mismatch()
    c.on(SimpleNamespace(get_type=lambda: "GPS_GLOBAL_ORIGIN", latitude=lat, longitude=lon,
                         altitude=alt - 10))
    assert c.origin_ok and c.origin_mismatch() is None


@pytest.mark.hazard("H-011", "H-013")
def test_vps_lost_at_the_facade_backs_off_along_the_normal_then_lands(grid):
    v, c, _ = companion(grid)
    H = (0.0, 12.0, 9.0)                                              # 3 m from the wall y = 15
    feed(v, c, H, 1.0)
    assert v.commands == [] and v.mode == "AUTO"
    c.vps_on = False
    feed(v, c, H, 1.0)
    assert v.commands[0] == (CMD_SET_EKF_SOURCE_SET, (2.0,))         # optical flow at once
    assert v.mode == "GUIDED" and c.phase == "backoff"
    vel = v.sent("set_position_target_local_ned")[-1]
    assert vel[3] == 9 and vel[8] < -0.4 and abs(vel[9]) < 0.05        # body frame: straight back
    assert vel[4] & 1024 and not vel[4] & 2048                        # yaw ignored, yaw RATE held
    feed(v, c, (0.0, 8.5, 9.0), 1.0)                                  # 6.5 m from the wall
    assert c.phase in ("brake", "land")
    feed(v, c, (0.0, 8.5, 9.0), 1.5)
    assert v.mode == "LAND"
    whats = [r[1] for r in c.nav.reactions]
    assert whats == ["VPS lost", "back off from the wall", "clear of the wall", "land"]


def test_vps_lost_in_transit_spends_the_budget_on_flow_then_stops(grid):
    v, c, _ = companion(grid, reaction={"drift_budget_s": 2.0})
    E = (0.0, 3.0, 9.0)                                              # the facade 12 m behind
    feed(v, c, E, 1.0, yaw_deg=180.0)
    c.vps_on = False
    feed(v, c, E, 1.5, yaw_deg=180.0)
    assert c.phase == "flow" and v.mode == "AUTO"                     # the mission goes on
    feed(v, c, E, 1.5, yaw_deg=180.0)
    assert [r[1] for r in c.nav.reactions][:2] == ["VPS lost", "stop (GUIDED)"]
    feed(v, c, E, 1.5, yaw_deg=180.0)
    assert v.mode == "LAND"


def test_a_short_gap_is_bridged_and_the_sources_come_back(grid):
    v, c, _ = companion(grid, reaction={"drift_budget_s": 5.0, "regain_s": 0.5})
    E = (0.0, 3.0, 9.0)
    feed(v, c, E, 1.0, yaw_deg=180.0)
    c.vps_on = False
    feed(v, c, E, 1.5, yaw_deg=180.0)
    c.vps_on = True
    feed(v, c, E, 1.5, yaw_deg=180.0)
    assert [cmd[1][0] for cmd in v.commands] == [2.0, 1.0] and v.mode == "AUTO"
    assert c.phase == "nominal"


def test_a_link_hiccup_is_not_a_vps_loss_but_a_stall_is(grid):
    v, c, _ = companion(grid, reaction={"drift_budget_s": 5.0})
    E = (0.0, 3.0, 9.0)
    feed(v, c, E, 1.0, yaw_deg=180.0)
    for _ in range(10):                 # 1 s without any SIM_STATE: VPS itself never said "no fix"
        v.t += 0.1
        c.tick()
    assert c.lost_t is None and v.commands == []
    for _ in range(15):                 # 2.5 s: nothing could be sent any more
        v.t += 0.1
        c.tick()
    whats = [r[1] for r in c.nav.reactions]
    assert whats == ["no truth from the simulator", "VPS lost"] and v.commands[0][1] == (2.0,)


def test_a_no_fix_zone_takes_vps_away(grid):
    v, c, _ = companion(grid, zones=[{"center": [0.0, 3.0], "radius_m": 2.0}])
    feed(v, c, (0.0, 0.0, 9.0), 1.0, tick=False)
    n = len(v.sent("odometry"))
    feed(v, c, (0.0, 3.0, 9.0), 1.0, tick=False)
    assert len(v.sent("odometry")) <= n + 1 and c.has_fix is False


@pytest.mark.hazard("H-014")
def test_gnss_far_from_vps_is_flagged_as_spoofed(grid):
    v = FakeVehicle()
    c = Companion(v, origin=ms.ORIGIN, sim=NavSim(gnss="monitor"), spec=ExtNavSpec(), flow=True,
                  grid=grid, speedup=1.0)
    feed(v, c, (0.0, 3.0, 9.0), 1.0, tick=False)

    def gps(enu):
        lat, lon, h = geodetic(EnuFrame(ms.ORIGIN), [enu])
        return SimpleNamespace(get_type=lambda: "GPS_RAW_INT", fix_type=3, lat=int(float(lat[0]) * 1e7),
                               lon=int(float(lon[0]) * 1e7), alt=int(float(h[0]) * 1000))

    for _ in range(5):
        c.on(gps((0.0, 3.5, 9.0)))
    assert c.nav.spoof_t is None
    for _ in range(3):
        v.t += 0.2
        c.on(gps((0.0, 23.0, 9.0)))
    assert c.nav.spoof_t == pytest.approx(v.t)


# ── the new oracles ──────────────────────────────────────────────────────
@pytest.mark.hazard("H-011")
def test_landing_next_to_the_wall_fails_the_wall_descent_oracle(gnss_plan, grid):
    rec = ss.fly(gnss_plan, until="hover")
    t_h = rec.t[-1]
    # GPS lost at H: LAND drifts towards the facade while descending (SIM-1, H-011).
    bad = ss.with_mode(rec, t_h, "LAND", path=[(0.0, 13.0, 6.0)])
    ctx = ss.context(gnss_plan, {"outcome": "abort", "reason": "gps_loss", "modes": ["LAND"]}, grid)
    v = check_wall_descent(bad, ctx)
    assert v.status == FAIL and v.metrics["min_wall_m"] < 2.5
    good = ss.with_mode(rec, t_h, "GUIDED", path=[(0.0, 8.5, 9.0)])
    assert check_wall_descent(good, ctx).status == PASS


def test_the_navigation_oracle(gnss_plan, grid):
    rec = ss.fly(gnss_plan)
    for k, p in enumerate(rec.enu):
        rec.est_enu[k] = (p[0] + 0.3, p[1], p[2])
    t0, t1 = rec.t[0], rec.t[-1]
    ctx = ss.context(gnss_plan, COMPLETE, grid)
    nav = NavContext(gnss=GnssUse.none, flow=True, fix=[(t0 - 5, True)], gnss_fixes=0, origin_ok=True)
    ctx.nav = nav
    v = check_navigation(rec, ctx)
    assert v.status == PASS and v.metrics["est_error_vps_p95_m"] == pytest.approx(0.3)
    assert "navigation" in [x.invariant for x in judge(rec, ctx, {"upload_verified": True})]
    for bad in (dict(gnss_fixes=3), dict(origin_ok=False), dict(spoof_injected_t=t0 + 5),
                dict(spoof_t=t0 + 1.0, spoof_injected_t=t0 + 5)):
        ctx.nav = NavContext(**{**nav.__dict__, **bad})
        assert check_navigation(rec, ctx).status == FAIL, bad
    for k, p in enumerate(rec.enu):
        rec.est_enu[k] = (p[0] + 3.0, p[1], p[2])
    ctx.nav = nav
    assert check_navigation(rec, ctx).status == FAIL
    ctx.nav = None
    assert check_navigation(rec, ctx).status == "n/a" and t1 > t0
