"""L0 tests of the L2 runner without SITL: the offline preflight that refuses a package before
anything is uploaded, the outcome decision, and the report. Every refusal is an expected
outcome of some scenario (§4.6: the correct refusal is a pass)."""

import json

import pytest

import mission_scenes as ms
import sim_scenarios as ss
from aerosplat.mission.export import format_waypoints, read_waypoints
from aerosplat.mission.models import Verdict as PlanVerdict
from aerosplat.sim.oracles import judge
from aerosplat.sim.record import Recording
from aerosplat.sim.runner import (MissionPackage, Refusal, RunResult, Runner, ScenarioError, decide,
                                  run_scenario)

COMPLETE = {"outcome": "complete", "reason": "delivered"}


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return ss.window_plan(tmp_path_factory.mktemp("plan"))


@pytest.fixture(scope="module")
def grid():
    return ms.street_grid()


@pytest.fixture
def mdir(tmp_path, plan):
    return ss.export(plan, tmp_path / "missions" / plan.id)


def refusal(mdir) -> Refusal:
    with pytest.raises(Refusal) as e:
        MissionPackage.load(mdir)
    return e.value


def runner(mdir, grid, expect=COMPLETE, **kw) -> Runner:
    return Runner(ss.scenario("t", expect, **kw), MissionPackage.load(mdir), grid, provider=None)


# ── the package ──────────────────────────────────────────────────────────
def test_a_consistent_export_loads_with_its_origin_from_the_plan(mdir, plan):
    pkg = MissionPackage.load(mdir)
    assert pkg.origin == ms.ORIGIN                       # from the .plan manifest
    assert pkg.items[0].frame == 0 and {it.frame for it in pkg.items[1:]} == {2, 3}
    assert len(pkg.fence_latlon) == len(plan.fence.polygon_enu) and len(pkg.rally) == 1
    assert pkg.role_seq("hover") == 6
    assert max(abs(a - b) for p, q in zip(pkg.fence_enu, plan.fence.polygon_enu)
               for a, b in zip(p, q)) < 0.05


@pytest.mark.hazard("H-003")
def test_a_package_with_the_8_1b_parameter_names_is_refused(mdir, grid):
    ss.legacy_parm(mdir)
    with pytest.raises(Refusal) as e:
        runner(mdir, grid).preflight()
    assert e.value.reason == "unknown_parameter"
    assert "RTL_ALT (did you mean RTL_ALT_M" in e.value.detail


@pytest.mark.hazard("H-002")
def test_amsl_heights_in_relative_items_are_an_altitude_frame_refusal(mdir):
    ss.amsl_heights(mdir)
    r = refusal(mdir)
    assert r.reason == "altitude_frame" and "+180.00 m" in r.detail


@pytest.mark.hazard("H-002")
def test_a_positional_item_in_an_absolute_frame_is_refused(mdir):
    items = read_waypoints((mdir / "mission.waypoints").read_text(encoding="utf-8"))
    bad = [items[0]] + [type(it)(**{**it.__dict__, "frame": 0}) if it.seq == 3 else it for it in items[1:]]
    (mdir / "mission.waypoints").write_text(format_waypoints(bad), encoding="utf-8")
    assert refusal(mdir).reason == "altitude_frame"


def test_waypoints_edited_after_export_are_refused(mdir):
    items = read_waypoints((mdir / "mission.waypoints").read_text(encoding="utf-8"))
    bad = [type(it)(**{**it.__dict__, "lat": it.lat + 1e-5}) if it.seq == 6 else it for it in items]
    (mdir / "mission.waypoints").write_text(format_waypoints(bad), encoding="utf-8")
    r = refusal(mdir)
    assert r.reason == "mission_mismatch" and "item 6 (hover)" in r.detail


def test_a_waypoint_edited_in_mission_json_off_its_checked_leg_is_refused(mdir, plan):
    """mission.json items moved, its legs (what was checked) not: refused before flight. The
    through-wall scenario moves both — that one only the flight oracle can catch."""
    items = [it.model_copy(update={"enu": ss.WALL_H}) if it.role == "hover" else it for it in plan.items]
    ss.export(plan.model_copy(update={"items": items}), mdir)
    r = refusal(mdir)
    assert r.reason == "mission_mismatch" and "off every checked leg" in r.detail
    ss.export(ss.through_wall(plan), mdir)
    MissionPackage.load(mdir)                            # consistent: loads


def test_a_plan_file_whose_fence_differs_is_refused(mdir):
    doc = json.loads((mdir / "mission.plan").read_text(encoding="utf-8"))
    doc["geoFence"]["polygons"][0]["polygon"][0][0] += 1e-4
    (mdir / "mission.plan").write_text(json.dumps(doc), encoding="utf-8")
    assert refusal(mdir).reason == "mission_mismatch"


def test_missing_files_and_missing_origin_are_refused(mdir):
    doc = json.loads((mdir / "mission.plan").read_text(encoding="utf-8"))
    del doc["aerosplat"]["origin"]
    (mdir / "mission.plan").write_text(json.dumps(doc), encoding="utf-8")
    r = refusal(mdir)
    assert r.reason == "package_incomplete" and "origin" in r.detail
    MissionPackage.load(mdir, origin=ms.ORIGIN)          # the scenario can give it
    (mdir / "mission.parm").unlink()
    assert refusal(mdir).reason == "package_incomplete"


def test_an_infeasible_plan_is_refused(mdir, plan, grid):
    ss.export(plan, mdir)
    pkg = MissionPackage.load(mdir)
    pkg.plan = plan.model_copy(update={"verdict": PlanVerdict.infeasible})
    r = Runner(ss.scenario("t", COMPLETE), pkg, grid, provider=None)
    with pytest.raises(Refusal, match="mission_infeasible"):
        r.preflight()


# ── the scenario against the package ────────────────────────────────────
def test_another_firmware_tag_is_refused(mdir, grid):
    with pytest.raises(Refusal) as e:
        runner(mdir, grid, ardupilot={"tag": "Copter-4.6.3"}).preflight()
    assert e.value.reason == "firmware_mismatch"


def test_preflight_reports_the_prescribed_response(mdir, grid):
    pre = runner(mdir, grid, {"outcome": "abort", "reason": "rc_loss"},
                 faults=[{"fault": "rc_loss", "after": "hover", "at_s": 3}]).preflight()
    assert pre["response"]["modes"][0] == "SMART_RTL" and pre["response"]["by"] == ["FS_THR_ENABLE"]
    assert pre["faults"] == ["no RC pulses (SIM_RC_FAIL 1)"] and pre["warnings"] == []


def test_a_fault_anchored_to_a_waypoint_the_mission_lacks_is_a_broken_scenario(mdir, grid):
    with pytest.raises(ScenarioError, match="no such waypoint"):
        runner(mdir, grid, {"outcome": "abort", "reason": "gps_loss"},
               faults=[{"fault": "gps_loss", "after": "entry", "at_s": 1}]).preflight()


def test_an_expected_mode_the_package_does_not_prescribe_is_a_broken_scenario(mdir, grid):
    with pytest.raises(ScenarioError, match="prescribes"):
        runner(mdir, grid, {"outcome": "abort", "reason": "gcs_loss", "modes": ["ALT_HOLD"]},
               faults=[{"fault": "gcs_loss", "at_s": 1}]).preflight()
    # LAND is the last fallback of "SmartRTL or RTL": a scenario may expect it.
    runner(mdir, grid, {"outcome": "abort", "reason": "gcs_loss", "modes": ["LAND"]},
           faults=[{"fault": "gcs_loss", "at_s": 1}]).preflight()


def test_an_abort_the_package_would_not_make_is_a_broken_scenario(mdir, grid):
    text = (mdir / "mission.parm").read_text(encoding="utf-8").replace("FS_OPTIONS 8", "FS_OPTIONS 9")
    (mdir / "mission.parm").write_text(text, encoding="utf-8")
    with pytest.raises(ScenarioError, match="carry on"):
        runner(mdir, grid, {"outcome": "abort", "reason": "rc_loss"},
               faults=[{"fault": "rc_loss", "at_s": 1}]).preflight()


# ── outcomes and reports ─────────────────────────────────────────────────
@pytest.mark.hazard("H-003")
def test_an_expected_refusal_is_a_pass_and_leaves_a_report(mdir, grid):
    ss.legacy_parm(mdir)
    sc = ss.scenario("legacy-parm", {"outcome": "refused", "reason": "unknown_parameter"})
    [r] = run_scenario(sc, mdir, grid, provider=None)
    assert r.passed and r.outcome == "refused" and "refused as expected" in r.summary
    doc = json.loads((mdir / "sim" / "legacy-parm" / "report.json").read_text(encoding="utf-8"))
    assert doc["passed"] and doc["reason"] == "unknown_parameter"
    assert "PASS" in (mdir / "sim" / "legacy-parm" / "report.html").read_text(encoding="utf-8")


def test_a_refusal_where_a_flight_was_expected_fails(mdir, grid):
    ss.amsl_heights(mdir)
    [r] = run_scenario(ss.scenario("x", COMPLETE), mdir, grid, provider=None)
    assert not r.passed and r.reason == "altitude_frame"


def test_the_wrong_refusal_fails(mdir, grid):
    ss.amsl_heights(mdir)
    [r] = run_scenario(ss.scenario("x", {"outcome": "refused", "reason": "unknown_parameter"}),
                       mdir, grid, provider=None)
    assert not r.passed and "expected refusal 'unknown_parameter'" in r.summary


def flown(plan, grid, expect, rec) -> RunResult:
    ctx = ss.context(plan, expect, grid)
    r = RunResult(ss.scenario("x", expect), 0, outcome="complete", recording=rec, grid=grid)
    r.verdicts = judge(rec, ctx, {"upload_verified": True})
    from aerosplat.sim.oracles import actual_outcome, clearance_series

    r.outcome = actual_outcome(rec, ctx)
    r.clearance = clearance_series(rec, ctx)
    decide(r)
    return r


def test_a_violation_scenario_passes_only_when_its_oracle_flags_it(plan, grid):
    viol = {"outcome": "violation", "reason": "clearance"}
    caught = flown(ss.through_wall(plan), grid, viol, ss.fly(ss.through_wall(plan)))
    assert caught.passed and "caught it" in caught.summary
    missed = flown(plan, grid, viol, ss.fly(plan))      # nothing wrong with this one
    assert not missed.passed and "clearance: expected fail, got pass" in missed.summary
    other = flown(ss.through_wall(plan), grid, COMPLETE, ss.fly(ss.through_wall(plan)))
    assert not other.passed and "clearance: expected pass, got fail" in other.summary


def test_report_of_a_flown_run(tmp_path, plan, grid):
    from aerosplat.sim.report import write_report

    ss.export(plan, tmp_path / "m")
    r = flown(plan, grid, COMPLETE, ss.fly(plan))
    r.package = MissionPackage.load(tmp_path / "m")
    r.timing = {"mission_start_sim_s": 10.0}
    html_path = write_report(r, tmp_path / "out")
    html = html_path.read_text(encoding="utf-8")
    assert r.passed and "✓ PASS" in html and html.count("<svg") == 3
    assert "кліренс" in html and "data:image/png;base64," in html
    doc = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert [v["invariant"] for v in doc["verdicts"]][0] == "clearance"
    assert doc["verdicts"][0]["expected"] == "pass" and doc["samples"] == len(r.recording)
    back = Recording.load(tmp_path / "out" / "trajectory.json")
    assert back.t == r.recording.t and back.events == r.recording.events
