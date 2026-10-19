"""SIM-3 (L2, ADR-0015 §5 stage 2): VPS in the loop on dataset_1910. `pytest -m sitl`.

The window B08/SW3/07/04 mission (tests/vps_mission.py) planned for the GNSS-free test bed
x500-pi5-imx290, with its corridor atlas and localizability rendered from run 20261019-105253;
flown in SITL at speedup 1 with the in-the-loop companion (aerosplat.sim.vpsloop): every VPS
tick renders the IMX290 frame at the TRUE pose (960x540, rolling shutter, `mild` camera effects)
and the localizer, with the autopilot's estimate as its prior, decides what is sent.

Needs the dataset (AEROSPLAT_PROJECTS for a checkout without it) and gsplat + CUDA
(tools\\gsplat_env.bat): without them the module skips. Environment: AEROSPLAT_SCENARIO_INSTANCE
(default 1). Rendered frames against an atlas rendered from the same splat: this is not the real
camera's domain gap (SIMULATION_AND_DELIVERY §4.13).
"""

import os

import pytest

import vps_mission as vm

pytestmark = pytest.mark.sitl

INSTANCE = int(os.environ.get("AEROSPLAT_SCENARIO_INSTANCE", "1"))
COMPLETE = {"outcome": "complete", "reason": "delivered"}
VPS_ABORT = {"outcome": "abort", "reason": "vps_loss", "modes": ["GUIDED"]}


@pytest.fixture(scope="module")
def world():
    why = vm.data_missing() or vm.gpu_missing()
    if why:
        pytest.skip(why)
    from aerosplat.vps.twin import EngineRenderer, SiteRun

    w = EngineRenderer(SiteRun.open(vm.RUN_DIR))
    yield w
    w.close()


@pytest.fixture(scope="module")
def package(world, tmp_path_factory):
    folder, plan, rep, grid = vm.package(tmp_path_factory.mktemp("sim3"), world)
    assert not plan.blocking, [f.detail for f in plan.blocking]
    return folder, plan, rep, grid


@pytest.fixture(scope="module")
def provider(package):
    try:
        import pymavlink  # noqa: F401
    except ImportError:
        pytest.fail("pymavlink is not installed: pip install -e .[sim]")
    from aerosplat.sim.runner import SitlProvider
    from aerosplat.sim.sitl import ENDPOINT_ENV, SitlError

    if os.environ.get(ENDPOINT_ENV):
        pytest.fail(f"VPS-in-the-loop scenarios boot SITL with their own parameters ({ENDPOINT_ENV})")
    try:
        with SitlProvider(instance=INSTANCE) as p:
            yield p
    except SitlError as e:
        pytest.fail(f"SITL: {e}")


@pytest.fixture
def fly(provider, package, world):
    import json

    from aerosplat.sim.runner import run_scenario
    from aerosplat.sim.scenario import Scenario

    folder, _, _, grid = package

    def run(name, expect, faults=(), vps=None, **kw):
        data = {"scenario": name, "site": "default/dataset_1910", "target": vm.TARGET_ID,
                "vehicle": vm.VEHICLE, "ardupilot": {"tag": "Copter-4.7.2"}, "speedup": 1,
                "timeout_s": 400, "expect": expect, "faults": list(faults),
                "nav": {"gnss": "none", "vps": {"run_dir": str(vm.RUN_DIR), **(vps or {})}}, **kw}
        sc = Scenario.model_validate(json.loads(json.dumps(data)))
        [r] = run_scenario(sc, folder, grid, provider=provider, world=world, log=print)
        report = folder / "sim" / name / "report.html"
        assert report.is_file()
        detail = "\n".join(f"  {v.invariant}: {v.status} — {v.detail}" for v in r.verdicts)
        print(f"{name}: {r.outcome} — {r.summary}\n{detail}")
        assert r.passed, f"{r.summary}\n{detail}\nreport: {report}"
        return r

    return run


def vps(r):
    return r.verdict("vps").metrics


def reactions(r):
    return [x[1] for x in r.preflight["nav"]["companion"]["reactions"]]


# ── (a) the delivery ─────────────────────────────────────────────────────
@pytest.mark.hazard("H-001", "H-011", "H-013")
def test_window_delivery_with_vps_in_the_loop_completes(fly, package):
    """Take-off on the anchor, the first fix in the take-off column (flown on flow until then),
    the window held on VPS fixes of rendered frames, the final descent on flow."""
    r = fly("sw3-window-vps-loop", COMPLETE)
    m = vps(r)
    assert m["false_fixes"] == 0 and m["rate_hz"] >= 2.0 and m["fix_rate"] >= 0.5
    assert m["err_m_p95"] < 0.5
    nav = r.verdict("navigation").metrics
    assert nav["gnss_fixes"] == 0 and nav["est_error_vps_p95_m"] < 1.5
    assert r.verdict("hover").metrics["hold_mean_horizontal_m"] < 1.0
    # lost in the take-off column (planned: the camera sees the ground), regained, landed on flow
    whats = reactions(r)
    assert whats[:2] == ["VPS lost", "VPS regained"] and whats[-1] == "landing on optical flow"


# ── (b) the camera goes blind: refusals -> flow -> the defined reaction ────
@pytest.mark.hazard("H-013")
def test_frames_blind_from_take_off_fall_back_to_flow_and_land_on_the_spot(fly):
    """Severe degradation (dusk, defocus, noise) from mission start: every frame refused, no fix
    after the anchor; optical flow carries the climb for the drift budget, then the companion
    stops (GUIDED) and lands — over the take-off spot, the only column checked for a landing."""
    # Stopped in the take-off column: no waypoint reached, so no altitude_frame to judge. The
    # reaction runs from lift-off (~2.5 s after the start, the anchor holds until then): 8 s
    # allowed on top of the drift budget instead of 5.
    expect = {**VPS_ABORT, "invariants": ["clearance", "fence", "failsafe", "completion",
                                          "wall_descent", "navigation", "vps"],
              "tolerances": {"react_s": 8}}
    r = fly("sw3-vps-blind-departure", expect,
            faults=[{"fault": "vps_degrade", "after": "start", "at_s": 0, "level": "severe"}])
    m = vps(r)
    assert m["sent"] == 0 and m["false_fixes"] == 0
    assert set(m["refusals"]) <= {"no_features", "too_few_matches", "too_few_inliers",
                                  "low_inlier_ratio", "poor_geometry", "pnp_failed"}
    assert reactions(r) == ["VPS lost", "stop (GUIDED)", "land"]
    assert r.verdict("completion").metrics["final"] is not None
    assert r.verdict("clearance").status == "pass"


@pytest.mark.hazard("H-011", "H-015")
@pytest.mark.xfail(strict=False, reason="H-015 / H-011, known hazard: VPS lost while holding 3 m from the "
                                        "facade. The forward beams read 6 m through the window opening, "
                                        "so the back-off ends at once; LAND in place then descends 1 m "
                                        "from the lower facade and into unknown space above the street "
                                        "(SITL 2026-10-20). No emergency landing site and no layer C "
                                        "yet (ADR-0015 §2.4, §8). Passes once a mitigation exists.")
def test_frames_degraded_at_the_facade_back_off_and_land_clear(fly):
    fly("sw3-vps-degraded-at-h", VPS_ABORT,
        faults=[{"fault": "vps_degrade", "after": "hover", "at_s": 5, "level": "severe"}])


# ── (c) another atlas than the package names ─────────────────────────────
@pytest.mark.hazard("H-013")
def test_another_atlas_on_the_companion_means_no_arming(fly, package, tmp_path):
    from aerosplat.vps.atlas import Atlas

    folder = package[0]
    other = Atlas.load(folder / "vps" / "atlas.npz")
    other.meta = {**other.meta, "note": "built again: same views, other metadata"}
    other.save(tmp_path / "other_atlas.npz")
    r = fly("sw3-vps-wrong-atlas", {"outcome": "refused", "reason": "atlas_mismatch"},
            vps={"atlas": str(tmp_path / "other_atlas.npz")})
    assert "is not the expected" in r.detail and "did not arm" in r.detail
    assert "VisOdom" in r.detail or "Position" in r.detail


# ── (d) a wrong prior ────────────────────────────────────────────────────
@pytest.mark.hazard("H-013")
def test_a_prior_15_m_off_is_refused_or_recovered_never_a_false_fix(fly):
    """For 2.5 s on the way out the localizer is handed a prior 15 m north of the autopilot's
    estimate: it refuses (or finds the true pose); no fix with a true error over 2 m is sent,
    flow bridges the gap and the delivery completes."""
    r = fly("sw3-vps-wrong-prior", COMPLETE,
            faults=[{"fault": "vps_prior_offset", "after": "over_entry", "at_s": 0, "duration_s": 2.5,
                     "offset_m": [15, 0]}])
    m = vps(r)
    assert m["false_fixes"] == 0 and m["prior_err_m_max"] >= 14.0
    inj = next(e for e in r.injected if e["fault"] == "vps_prior_offset")
    during = [x for x in r.preflight["nav"]["companion"]["vps"]["records"]
              if inj["on"] + 0.1 <= x["t"] <= inj["off"]]      # captured with the offset set
    assert during and all(x["prior_err_m"] >= 14.0 for x in during)
    assert all(x["err_m"] <= 2.0 for x in during if x["sent"])
    print("during the offset:", [(x["t"], x["reason"] or "fix") for x in during])
