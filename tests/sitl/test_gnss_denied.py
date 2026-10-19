"""SIM-2n (L2, ADR-0015 stage 1): the window mission flown with NO GNSS, on ExternalNav from the
companion emulator (aerosplat.sim.extnav: SIM_STATE truth + a VPS error model, as MAVLink
ODOMETRY) and optical flow as the degraded source. `pytest -m sitl`.

Same street, same mission geometry as tests/sitl/test_scenarios.py (take off to 9 m, E (0, 3, 9),
approach H (0, 12, 9) facing the facade at y = 15, hold 10 s, back, land), planned with the
GNSS-free navigation profile and flown by the `sitl-quad-extnav` vehicle (SITL flow sensor and
downward rangefinder). SITL boots with GPS1_TYPE 0 and no simulated GNSS signal at all.

The forward rangefinders the H-011 reaction uses are a stand-in: rays into the twin's occupancy
from the true pose (SIM-2 physics will provide rng_* instead).

Environment: AEROSPLAT_SCENARIO_INSTANCE (default 1), AEROSPLAT_SCENARIO_SPEEDUP (default 3).
"""

import os

import pytest

import mission_scenes as ms
import sim_scenarios as ss

pytestmark = pytest.mark.sitl

SPEEDUP = float(os.environ.get("AEROSPLAT_SCENARIO_SPEEDUP", "3"))
INSTANCE = int(os.environ.get("AEROSPLAT_SCENARIO_INSTANCE", "1"))
VEHICLE = "sitl-quad-extnav"


@pytest.fixture(scope="module")
def provider():
    try:
        import pymavlink  # noqa: F401
    except ImportError:
        pytest.fail("pymavlink is not installed: pip install -e .[sim]")
    from aerosplat.sim.runner import SitlProvider
    from aerosplat.sim.sitl import ENDPOINT_ENV, SitlError

    if os.environ.get(ENDPOINT_ENV):
        pytest.fail("GNSS-free scenarios boot SITL with their own parameters: they cannot fly a "
                    f"given endpoint ({ENDPOINT_ENV})")
    try:
        with SitlProvider(instance=INSTANCE) as p:
            yield p
    except SitlError as e:
        pytest.fail(f"SITL: {e}")


@pytest.fixture(scope="module")
def street():
    return ms.street_grid()


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return ss.nav_plan(tmp_path_factory.mktemp("plan"), gnss="none")


@pytest.fixture(scope="module")
def monitor_plan(tmp_path_factory):
    return ss.nav_plan(tmp_path_factory.mktemp("plan-monitor"), gnss="monitor")


@pytest.fixture
def fly(provider, street, tmp_path):
    from aerosplat.sim.runner import run_scenario

    def run(name, plan, expect, faults=(), nav=None, **kw):
        mdir = ss.export(plan, tmp_path / "missions" / plan.id)
        nav = {"gnss": plan.profile.nav.gnss.value, **(nav or {})}
        sc = ss.scenario(name, expect, faults=list(faults), speedup=SPEEDUP, vehicle=VEHICLE,
                         nav=nav, **kw)
        [r] = run_scenario(sc, mdir, street, provider=provider, log=print)
        report = mdir / "sim" / name / "report.html"
        assert report.is_file() and (report.parent / "report.json").is_file()
        detail = "\n".join(f"  {v.invariant}: {v.status} — {v.detail}" for v in r.verdicts)
        print(f"{name}: {r.outcome} — {r.summary}\n{detail}")
        assert r.passed, f"{r.summary}\n{detail}\nreport: {report}"
        return r

    return run


def metric(r, invariant, key):
    return r.verdict(invariant).metrics[key]


def companion(r):
    return r.preflight["nav"]["companion"]


# ── (a) delivery without GNSS ────────────────────────────────────────────
@pytest.mark.hazard("H-001", "H-011", "H-013")
def test_window_delivery_without_gnss_completes(fly, plan):
    r = fly("street-window-gnss-none", plan, {"outcome": "complete", "reason": "delivered"},
            nav={"bias_sigma_m": 0.2, "bias_tau_s": 30})
    assert r.outcome == "complete"
    nav = r.verdict("navigation").metrics
    assert nav["gnss_fixes"] == 0 and nav["origin_ok"]
    assert nav["est_error_vps_p95_m"] < 1.5
    assert metric(r, "hover", "hold_mean_horizontal_m") < 1.0      # the hover tolerance
    assert metric(r, "clearance", "min_clearance_m") >= 2.0
    assert r.preflight["nav"]["origin_verified"]["lat"] == pytest.approx(ms.ORIGIN.lat, abs=1e-7)
    assert "GPS1_TYPE" in r.preflight["nav"]["boot_verified"]
    assert not [w for w in r.warnings if "differs from the" in w], r.warnings


# ── (b) VPS lost in transit ──────────────────────────────────────────────
@pytest.mark.hazard("H-013")
def test_vps_loss_in_transit_falls_back_to_flow_then_lands(fly, plan):
    """VPS gone for good on the way back, 4 s after leaving H (the facade behind, out of the
    forward rangefinders and beyond the guard distance): EKF3 source set 2 (optical flow) at
    once, the mission goes on for the drift budget on flow, then the companion stops it and
    lands — clear of everything."""
    r = fly("street-window-vps-loss-transit", plan,
            {"outcome": "abort", "reason": "vps_loss", "modes": ["GUIDED"]},
            faults=[{"fault": "vps_loss", "after": "over_exit", "at_s": 4}],
            nav={"reaction": {"drift_budget_s": 3}})
    c = companion(r)
    assert [s[1] for s in c["source_sets"]] == [2] and c["source_sets"][0][2] == 0
    whats = [x[1] for x in c["reactions"]]
    assert whats == ["VPS lost", "stop (GUIDED)", "land"], whats
    f = r.verdict("failsafe").metrics
    assert f["first_mode"] == "GUIDED" and 3.0 <= f["react_s"] < 5.0   # budget + detection
    assert r.verdict("wall_descent").status == "pass"


@pytest.mark.hazard("H-013")
def test_short_vps_loss_is_bridged_by_flow_and_the_mission_completes(fly, plan):
    r = fly("street-window-vps-gap", plan, {"outcome": "complete", "reason": "delivered"},
            faults=[{"fault": "vps_loss", "after": "over_exit", "at_s": 4, "duration_s": 2}],
            nav={"reaction": {"drift_budget_s": 6}})
    sets = [s[1] for s in companion(r)["source_sets"]]
    assert sets == [2, 1], sets
    assert "est_error_flow_max_m" in r.verdict("navigation").metrics


# ── (c) H-011: VPS lost while holding 3 m from the facade ────────────────
@pytest.mark.hazard("H-011", "H-013")
def test_vps_loss_at_the_facade_backs_off_before_landing(fly, plan):
    """The case SIM-1 left as xfail (GPS lost at H: LAND drifted into the wall). Here the
    companion switches to flow, backs away along the wall's normal (forward rangefinders,
    a stand-in) to 6 m, and only then lands."""
    r = fly("street-window-vps-loss-at-h", plan,
            {"outcome": "abort", "reason": "vps_loss", "modes": ["GUIDED"]},
            faults=[{"fault": "vps_loss", "after": "hover", "at_s": 12}])
    whats = [x[1] for x in companion(r)["reactions"]]
    assert whats[:2] == ["VPS lost", "back off from the wall"], whats
    assert "clear of the wall" in whats and whats[-1] == "land"
    w = r.verdict("wall_descent").metrics
    assert w["descending_samples"] > 0 and (w["min_wall_m"] is None or w["min_wall_m"] >= 4.0)
    assert r.verdict("clearance").status == "pass"
    assert r.verdict("completion").metrics["final"][1] <= 15.0 - 4.0     # landed ≥ 4 m from the wall


# ── (d) negative: no arming without a VPS fix or without the origin ──────
@pytest.mark.hazard("H-013")
def test_no_arming_without_an_extnav_fix(fly, plan):
    r = fly("street-window-no-extnav", plan, {"outcome": "refused", "reason": "arming_refused"},
            nav={"extnav": False})
    assert "VisOdom" in r.detail or "Position" in r.detail, r.detail


@pytest.mark.hazard("H-002", "H-013")
def test_no_arming_without_the_site_origin(fly, plan):
    r = fly("street-window-no-origin", plan, {"outcome": "refused", "reason": "arming_refused"},
            nav={"origin": False})
    assert "position" in r.detail.lower(), r.detail


# ── (e) GNSS as a monitor: a spoof is flagged and changes nothing ────────
@pytest.mark.hazard("H-014")
def test_gnss_spoof_is_flagged_and_ignored(fly, monitor_plan):
    r = fly("street-window-gnss-spoof", monitor_plan, {"outcome": "complete", "reason": "delivered"},
            faults=[{"fault": "gps_spoof", "after": "hover", "at_s": 5, "offset_m": [20, 0]}])
    nav = r.verdict("navigation").metrics
    assert nav["spoof_detect_s"] < 3.0
    assert nav["est_error_vps_p95_m"] < 1.5      # the EKF never followed the spoofed GNSS
