"""SIM-1 (L2): missions exported by aerosplat.mission flown in the pinned ArduCopter SITL and
judged on the TRUE state (SIM_STATE) against the twin. `pytest -m sitl`.

All scenarios fly the window mission of the synthetic street (tests/sim_scenarios.py): take off
to 9 m, stop at E (0, 3, 9), approach H (0, 12, 9) at 1 m/s facing the window, hold 10 s, come
back, land. Each run starts a fresh SITL instance (wiped parameters) with home = the scene's
origin, uploads fence, rally points, mission and the .parm, and writes report.html + report.json
next to the export (<mission>/sim/<scenario>/).

A negative scenario passes when the run ends the way the package says it must — a refusal, an
abort with the prescribed failsafe mode, or the oracle flagging what was wrong (§4.6).

Environment: AEROSPLAT_SCENARIO_INSTANCE (SITL instance, default 1, so that a SITL on instance 0
— Mission Planner, the smoke test — is left alone), AEROSPLAT_SCENARIO_SPEEDUP (default 3).
"""

import os

import pytest

import mission_scenes as ms
import sim_scenarios as ss

pytestmark = pytest.mark.sitl

SPEEDUP = float(os.environ.get("AEROSPLAT_SCENARIO_SPEEDUP", "3"))
INSTANCE = int(os.environ.get("AEROSPLAT_SCENARIO_INSTANCE", "1"))


@pytest.fixture(scope="module")
def provider():
    try:
        import pymavlink  # noqa: F401
    except ImportError:
        pytest.fail("pymavlink is not installed: pip install -e .[sim]")
    from aerosplat.sim.runner import SitlProvider
    from aerosplat.sim.sitl import ENDPOINT_ENV, SitlError

    try:
        with SitlProvider(instance=INSTANCE, endpoint=os.environ.get(ENDPOINT_ENV)) as p:
            yield p
    except SitlError as e:
        pytest.fail(f"SITL: {e}")


@pytest.fixture(scope="module")
def street():
    return ms.street_grid()


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return ss.window_plan(tmp_path_factory.mktemp("plan"))


@pytest.fixture
def fly(provider, street, tmp_path):
    from aerosplat.sim.runner import run_scenario

    def run(name, plan, expect, faults=(), **kw):
        mdir = ss.export(plan, tmp_path / "missions" / plan.id)
        sc = ss.scenario(name, expect, faults=list(faults), speedup=SPEEDUP, **kw)
        [r] = run_scenario(sc, mdir, street, provider=provider, log=lambda *a: None)
        report = mdir / "sim" / name / "report.html"
        assert report.is_file() and (report.parent / "report.json").is_file()
        detail = "\n".join(f"  {v.invariant}: {v.status} — {v.detail}" for v in r.verdicts)
        assert r.passed, f"{r.summary}\n{detail}\nreport: {report}"
        return r

    return run


def metric(r, invariant, key):
    return r.verdict(invariant).metrics[key]


# ── positive ─────────────────────────────────────────────────────────────
@pytest.mark.hazard("H-001", "H-002", "H-003", "H-009")
def test_window_delivery_completes_and_every_invariant_holds(fly, plan):
    r = fly("street-window-delivered", plan, {"outcome": "complete", "reason": "delivered"})
    assert r.outcome == "complete"
    assert r.firmware["git"] == r.firmware["pin"]
    # The live parameter table is the snapshot the exporter and the faults are checked against.
    assert not [w for w in r.warnings if "differs from the" in w], r.warnings
    assert metric(r, "clearance", "min_clearance_m") >= 2.0
    assert metric(r, "hover", "hold_mean_horizontal_m") < 0.5
    # The stop at E makes the approach fly at the planned 1 m/s (2.4 m/s without it).
    assert metric(r, "hover", "approach_max_speed_mps") <= 1.3
    assert metric(r, "altitude_frame", "worst_dz_m") < 0.5
    assert r.preflight["upload_verified"] and r.preflight["frames"] == [0, 3]


def test_yaw_command_before_h_acts_on_the_approach_leg(fly, plan):
    """8.1b open item: CONDITION_YAW sits after the waypoint at E. Commanded 90° on a northbound
    approach, the vehicle must already face east when it arrives at H."""
    r = fly("street-window-yaw-east", ss.yaw_to(plan, 90.0),
            {"outcome": "complete", "reason": "delivered"})
    m = r.verdict("heading").metrics
    assert m["commanded_deg"] == 90.0 and m["error_on_arrival_deg"] < 10
    assert m["faced_before_arrival_s"] > 0


# ── negative ─────────────────────────────────────────────────────────────
@pytest.mark.hazard("H-001")
def test_corridor_through_the_wall_is_flagged_by_the_clearance_oracle(fly, plan):
    """The plan itself is wrong (H moved 2 m into the building, files consistent): SITL has no
    walls, so the flight 'succeeds' — and the oracle, judging the truth against the twin, fails it."""
    r = fly("street-window-through-wall", ss.through_wall(plan),
            {"outcome": "violation", "reason": "clearance"})
    v = r.verdict("clearance")
    assert v.status == "fail" and v.metrics["entered"]["state"] == "occupied"
    assert v.metrics["entered"]["at"][1] == pytest.approx(15.0, abs=0.5)


@pytest.mark.hazard("H-011")
def test_gps_loss_lands_as_the_package_prescribes(fly, plan):
    """GPS lost on the way back from the window. The package says LAND (FS_EKF_ACTION 1; the
    dead-reckoning failsafe cannot fire on this copter). The EKF needs seconds to give up, and
    the mission goes on meanwhile — measured and reported, inside the allowed delay."""
    r = fly("street-window-gps-loss", plan, {"outcome": "abort", "reason": "gps_loss", "modes": ["LAND"]},
            faults=[{"fault": "gps_loss", "after": "over_exit", "at_s": 1}])
    f = r.verdict("failsafe").metrics
    assert f["first_mode"] == "LAND" and 1.0 < f["react_s"] < 15.0
    assert r.injected[0]["params"] == [["SIM_GPS1_ENABLE", 0.0]]


@pytest.mark.hazard("H-011")
@pytest.mark.xfail(strict=False, reason="H-011, known hazard: GPS lost while holding 3 m from the "
                                        "facade -> LAND after ~8 s on IMU alone -> the vehicle drifts "
                                        "~0.5 m/s and descends into the wall (SITL 2026-10-19: 7.6 m, "
                                        "hit the wall at 4.2 m). Passes once a mitigation exists.")
def test_gps_loss_at_the_facade_lands_clear_of_it(fly, plan):
    fly("street-window-gps-loss-at-h", plan, {"outcome": "abort", "reason": "gps_loss", "modes": ["LAND"]},
        faults=[{"fault": "gps_loss", "after": "hover", "at_s": 12}])


@pytest.mark.hazard("H-009")
def test_fence_breach_returns_by_smart_rtl(fly, plan):
    r = fly("street-window-fence-breach", ss.outside_fence(plan),
            {"outcome": "abort", "reason": "fence_breach", "modes": ["SMART_RTL"]})
    assert r.verdict("fence").status == "fail"                        # the breach is the trigger
    assert r.verdict("fence").metrics["max_outside_m"] < 10.0         # ... and it was stopped
    assert r.verdict("failsafe").metrics["first_mode"] == "SMART_RTL"
    assert r.verdict("completion").metrics["final_from_home_m"] < 1.5


@pytest.mark.hazard("H-008")
def test_rc_loss_returns_by_smart_rtl(fly, plan):
    r = fly("street-window-rc-loss", plan, {"outcome": "abort", "reason": "rc_loss", "modes": ["SMART_RTL"]},
            faults=[{"fault": "rc_loss", "after": "hover", "at_s": 12}])
    assert r.injected[0]["params"] == [["SIM_RC_FAIL", 1.0]]
    assert r.verdict("completion").metrics["final_from_home_m"] < 1.5


@pytest.mark.hazard("H-008")
def test_gcs_loss_returns_by_smart_rtl_after_its_timeout(fly, plan):
    r = fly("street-window-gcs-loss", plan, {"outcome": "abort", "reason": "gcs_loss", "modes": ["SMART_RTL"]},
            faults=[{"fault": "gcs_loss", "after": "hover", "at_s": 2}])
    f = r.verdict("failsafe").metrics
    assert 4.0 < f["react_s"] < 10.0                                  # FS_GCS_TIMEOUT 5 s


@pytest.mark.hazard("H-012")
def test_low_battery_returns_by_smart_rtl(fly, plan):
    r = fly("street-window-battery-low", plan,
            {"outcome": "abort", "reason": "battery_low", "modes": ["SMART_RTL"]},
            faults=[{"fault": "battery_sag", "level": "low", "after": "start", "at_s": 1}])
    f = r.verdict("failsafe").metrics
    assert 9.0 < f["react_s"] < 15.0                                  # BATT_LOW_TIMER 10 s
