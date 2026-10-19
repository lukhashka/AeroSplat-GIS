"""L0 tests of the L2 oracles on synthetic TRUE trajectories over the street scene (no SITL).

`sim_scenarios.fly` flies the window mission exactly: home (0, 0, 0) -> 9 m -> E (0, 3, 9),
2 s stop -> H (0, 12, 9) at 1 m/s facing north -> hold 10 s -> back -> land. The required
clearance is r_drone 0.5 + min(m 1.5, transit margin 2.0) = 2.0 m; H is 3 m from the wall.
Every negative test changes one thing about that flight and expects the matching oracle — and
only it — to fail.
"""

import numpy as np
import pytest

import mission_scenes as ms
import sim_scenarios as ss
from aerosplat.sim.oracles import (FAIL, NA, PASS, actual_outcome, check_altitude_frame,
                                   check_clearance, check_completion, check_failsafe, check_fence,
                                   check_heading, check_hover, clearance_series, expected_status,
                                   fence_breach_time, judge)
from aerosplat.sim.faults import response
from aerosplat.sim.record import Recording
from aerosplat.sim.scenario import Expect
from aerosplat.surface.models import UNKNOWN

COMPLETE = {"outcome": "complete", "reason": "delivered"}


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return ss.window_plan(tmp_path_factory.mktemp("plan"))


@pytest.fixture(scope="module")
def grid():
    return ms.street_grid()


@pytest.fixture(scope="module")
def nominal(plan):
    return ss.fly(plan)


def moved(rec: Recording, fn) -> Recording:
    """A copy of `rec` with every true position p replaced by fn(t, p)."""
    out = Recording.from_dict(rec.to_dict())
    out.enu = [tuple(float(v) for v in fn(t, np.asarray(p, float))) for t, p in zip(rec.t, rec.enu)]
    return out


def leg(rec: Recording, plan, role: str) -> np.ndarray:
    seq = next(k for k, it in enumerate(plan.items, start=1) if it.role == role)
    return np.asarray(rec.seq) == seq


# ── the nominal flight passes everything ─────────────────────────────────
@pytest.mark.hazard("H-001", "H-009")
def test_nominal_flight_passes_every_oracle(plan, grid, nominal):
    ctx = ss.context(plan, COMPLETE, grid)
    verdicts = judge(nominal, ctx, {"upload_verified": True, "frames": [0, 2, 3]})
    assert [v.invariant for v in verdicts] == ["clearance", "fence", "hover", "heading",
                                               "altitude_frame", "failsafe", "completion",
                                               "wall_descent"]
    assert all(v.status == PASS for v in verdicts), [(v.invariant, v.detail) for v in verdicts]
    assert actual_outcome(nominal, ctx) == "complete"
    c = verdicts[0].metrics
    assert c["required_m"] == pytest.approx(2.0)
    assert c["min_target_distance_m"] == pytest.approx(3.0, abs=0.05)   # H: 3 m from the wall
    assert verdicts[2].metrics["approach_max_speed_mps"] == pytest.approx(1.0)


def test_ground_under_home_and_target_wall_do_not_count_where_touched(plan, grid, nominal):
    """Take-off and landing touch the ground, the hold is 3 m from the wall it faces: neither
    may read as a clearance violation, but the wall distance is kept in the record."""
    s = clearance_series(nominal, ss.context(plan, COMPLETE, grid))
    assert s.clearance.min() >= s.required
    assert np.nanmin(s.target_distance) == pytest.approx(3.0, abs=0.05)
    assert np.isnan(s.target_distance[0])            # on the ground at home: not in the zone


def test_standing_in_the_grounds_relief_is_not_entering_it(plan, grid, nominal):
    """dataset_v6: at rest on the pad the vehicle sits 0.1 m above a ground that the 0.5 m grid
    puts a little higher in the next column. That is the surface it stands on; 1 m below the
    ground under home is not."""
    on_pad = moved(nominal, lambda t, p: p + [0, 0, -0.2] if p[2] < 0.5 else p)
    assert check_clearance(on_pad, ss.context(plan, COMPLETE, grid)).status == PASS
    sunk = moved(nominal, lambda t, p: p + [0, 0, -1.0] if p[2] < 0.5 else p)
    v = check_clearance(sunk, ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and "enters occupied space" in v.detail


# ── clearance: occupied, unknown, too close ──────────────────────────────
@pytest.mark.hazard("H-001")
def test_trajectory_into_the_wall_is_flagged(plan, grid):
    rec = ss.fly(ss.through_wall(plan))
    v = check_clearance(rec, ss.context(ss.through_wall(plan), COMPLETE, grid))
    assert v.status == FAIL and "enters occupied space" in v.detail
    assert v.metrics["entered"]["at"][1] == pytest.approx(15.0, abs=0.3)   # the wall plane


@pytest.mark.hazard("H-001")
def test_closer_to_the_wall_than_the_class_allows_is_flagged(plan, grid, nominal):
    """Holding 1.5 m from the wall instead of 3 m: still in free air, but closer than
    r_drone + m = 2 m to the target surface."""
    hover = leg(nominal, plan, "hover")
    rec = moved(nominal, lambda t, p: p + np.array([0, 1.5, 0]) * hover[nominal.t.index(t)])
    v = check_clearance(rec, ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and "target surface" in v.detail
    assert v.metrics["min_target_distance_m"] == pytest.approx(1.5, abs=0.05)


@pytest.mark.hazard("H-001")
def test_unknown_space_counts_as_occupied(plan):
    """A block nobody has observed, 1 m beside the transit at E: the twin does not say it is
    solid, and that is exactly why it counts."""
    state = ms.base_state()
    ms.put(state, (1.5, 4.0, 2.0, 4.0, 0.0, 12.0), UNKNOWN)
    grid = ms.grid_from_state(state)
    v = check_clearance(ss.fly(plan), ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and v.metrics["min_clearance_m"] < 2.0
    assert v.metrics["min_clearance_m"] == pytest.approx(1.5, abs=0.3)


def test_flying_through_unknown_space_is_flagged_as_unknown(plan):
    state = ms.base_state()
    ms.put(state, (-3.0, 3.0, 6.0, 8.0, 0.0, 12.0), UNKNOWN)   # across the approach leg
    v = check_clearance(ss.fly(plan), ss.context(plan, COMPLETE, ms.grid_from_state(state)))
    assert v.status == FAIL and "enters unknown space" in v.detail


# ── fence ────────────────────────────────────────────────────────────────
@pytest.mark.hazard("H-009")
def test_waypoint_outside_the_fence_is_a_breach(plan, grid):
    p = ss.outside_fence(plan)
    rec = ss.fly(p)
    ctx = ss.context(p, COMPLETE, grid)
    v = check_fence(rec, ctx)
    assert v.status == FAIL and v.metrics["max_outside_m"] == pytest.approx(25 - 10.6, abs=0.3)
    assert fence_breach_time(rec, ctx) == pytest.approx(v.metrics["first_breach_t"])


@pytest.mark.hazard("H-009")
def test_climbing_above_the_fence_altitude_is_a_breach(plan, grid, nominal):
    ctx = ss.context(plan, COMPLETE, grid)
    top = ctx.params["FENCE_ALT_MAX"]
    hover = leg(nominal, plan, "hover")
    rec = moved(nominal, lambda t, p: p + np.array([0, 0, top]) * hover[nominal.t.index(t)])
    v = check_fence(rec, ctx)
    assert v.status == FAIL and v.metrics["max_outside_m"] == pytest.approx(9.0, abs=0.2)


# ── hover and heading ────────────────────────────────────────────────────
def test_stopping_short_of_h_fails_hover(plan, grid, nominal):
    hover = leg(nominal, plan, "hover")
    rec = moved(nominal, lambda t, p: np.minimum(p, [np.inf, 10.0, np.inf])
                if hover[nominal.t.index(t)] else p)
    v = check_hover(rec, ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and v.metrics["min_horizontal_m"] == pytest.approx(2.0, abs=0.05)


def test_too_short_a_hold_fails_hover(plan, grid):
    short = plan.model_copy(update={"items": [it.model_copy(update={"param1": 1.0})
                                              if it.role == "hover" else it for it in plan.items]})
    v = check_hover(ss.fly(short), ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and v.metrics["held_s"] < 3.0


def test_approach_faster_than_planned_fails_hover(plan, grid, nominal):
    """SITL showed it: without the stop at E the 1 m/s approach flew at 2.4 m/s."""
    rec = Recording.from_dict(nominal.to_dict())
    hover = leg(nominal, plan, "hover")
    rec.vel_ned = [(2.4, 0.0, 0.0) if h else v for h, v in zip(hover, rec.vel_ned)]
    v = check_hover(rec, ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and "2.40 m/s" in v.detail


def test_heading_command_must_act_before_arriving(plan, grid):
    """Commanded 90°: a vehicle that arrives facing its direction of travel (0°) has not executed
    the CONDITION_YAW on the approach leg."""
    p90 = ss.yaw_to(plan, 90.0)
    ctx = ss.context(p90, COMPLETE, grid)
    assert check_heading(ss.fly(p90), ctx).status == PASS
    v = check_heading(ss.fly(plan), ctx)          # flies with the original 0° heading
    assert v.status == FAIL and "did not act on the approach leg" in v.detail


def test_heading_is_not_applicable_without_a_yaw_command(tmp_path, grid):
    ap = ms.roof_pad_ap()
    target, _, _ = ms.confirmed(tmp_path, ap)
    from aerosplat.mission.planner import plan_mission

    pad = plan_mission(target=target, ap=ap, grid=grid, occupancy=ms.occupancy_ref(),
                       origin=ms.ORIGIN, by="op:t", profile=ss.PROFILE)
    assert check_heading(ss.fly(pad), ss.context(pad, COMPLETE, grid)).status == NA


# ── altitude frame ───────────────────────────────────────────────────────
@pytest.mark.hazard("H-002")
def test_flying_heights_off_by_a_misplaced_home_is_flagged(plan, grid, nominal):
    """Every waypoint 3 m higher than planned: what a home 3 m below the plan's would do to
    relative altitudes (and AMSL numbers read as relative would do a hundredfold)."""
    rec = moved(nominal, lambda t, p: p + [0, 0, 3.0] if p[2] > 0.5 else p)
    v = check_altitude_frame(rec, ss.context(plan, COMPLETE, grid), {"upload_verified": True})
    assert v.status == FAIL and "+3.00 m" in v.detail


@pytest.mark.hazard("H-002")
def test_arming_away_from_the_plans_home_is_flagged(plan, grid, nominal):
    rec = moved(nominal, lambda t, p: p + [4.0, 0, 0])
    v = check_altitude_frame(rec, ss.context(plan, COMPLETE, grid), {"upload_verified": True})
    assert v.status == FAIL and "from the plan's home" in v.detail


def test_upload_not_verified_fails_the_altitude_frame(plan, grid, nominal):
    v = check_altitude_frame(nominal, ss.context(plan, COMPLETE, grid), {"upload_verified": False})
    assert v.status == FAIL


# ── failsafe ─────────────────────────────────────────────────────────────
GPS_ABORT = {"outcome": "abort", "reason": "gps_loss", "modes": ["LAND"]}


def gps_ctx(plan, grid, trigger_t):
    from aerosplat.mission.export import parameters

    parm = {n: float(v) for n, v, _ in parameters(plan)}
    ctx = ss.context(plan, GPS_ABORT, grid, trigger_t=trigger_t)
    ctx.response = response("gps_loss", parm, ctx.params)
    return ctx


@pytest.mark.hazard("H-011")
def test_expected_failsafe_mode_in_time_passes(plan, grid, nominal):
    """GPS lost in the hold at H, LAND 8 s later (what SITL does: the EKF needs that long)."""
    trig = float(nominal.t[int(np.argmax(leg(nominal, plan, "hover")))]) + 6.0
    rec = ss.with_mode(nominal, trig + 8.0, "LAND")
    v = check_failsafe(rec, gps_ctx(plan, grid, trig))
    assert v.status == PASS, v.detail
    assert v.metrics["react_s"] == pytest.approx(8.0, abs=0.11)
    assert v.metrics["flown_before_reaction_m"] > 0      # it went on along the mission meanwhile


@pytest.mark.hazard("H-011")
@pytest.mark.parametrize("case", ["wrong_mode", "too_late", "before_trigger", "none", "back_to_auto"])
def test_wrong_failsafe_reactions_fail(plan, grid, nominal, case):
    trig = float(nominal.t[int(np.argmax(leg(nominal, plan, "hover")))]) + 6.0
    ctx = gps_ctx(plan, grid, trig)
    if case == "wrong_mode":
        rec = ss.with_mode(nominal, trig + 2.0, "RTL")
    elif case == "too_late":
        rec = ss.with_mode(nominal, trig + 16.0, "LAND")
    elif case == "before_trigger":
        rec = ss.with_mode(nominal, trig - 2.0, "LAND")
    elif case == "none":
        rec = nominal
    else:
        rec = ss.with_mode(nominal, trig + 2.0, "LAND")
        rec.event(trig + 4.0, "mode", "AUTO", previous="LAND")
    v = check_failsafe(rec, ctx)
    assert v.status == FAIL, (case, v.detail)


@pytest.mark.hazard("H-008")
def test_unexpected_failsafe_in_a_delivery_fails(plan, grid, nominal):
    rec = ss.with_mode(nominal, 40.0, "SMART_RTL", path=[(0, 0, 9)])
    ctx = ss.context(plan, COMPLETE, grid)
    assert check_failsafe(rec, ctx).status == FAIL
    assert actual_outcome(rec, ctx) == "abort"


# ── completion ───────────────────────────────────────────────────────────
def test_landing_away_from_home_fails_completion(plan, grid, nominal):
    land = leg(nominal, plan, "land")
    rec = moved(nominal, lambda t, p: p + [3.0, 0, 0] if land[nominal.t.index(t)] or
                not nominal.armed[nominal.t.index(t)] else p)
    v = check_completion(rec, ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and "from home" in v.detail


def test_still_airborne_at_the_timeout_fails_completion(plan, grid):
    rec = ss.fly(plan, until="hover")
    v = check_completion(rec, ss.context(plan, COMPLETE, grid))
    assert v.status == FAIL and "timeout" in v.detail


def test_abort_that_completed_the_mission_fails(plan, grid, nominal):
    v = check_completion(nominal, ss.context(plan, GPS_ABORT, grid))
    assert v.status == FAIL and "should have been aborted" in v.detail


def test_rtl_abort_must_come_home(plan, grid, nominal):
    trig = 30.0
    rec = ss.with_mode(nominal, trig, "SMART_RTL", path=[(0, 3, 9), (5, 0, 9)])  # lands 5 m off
    v = check_completion(rec, ss.context(plan, GPS_ABORT, grid))
    assert v.status == FAIL and "from home" in v.detail
    ok = ss.with_mode(nominal, trig, "SMART_RTL", path=[(0, 3, 9), (0, 0, 9)])
    assert check_completion(ok, ss.context(plan, GPS_ABORT, grid)).status == PASS


# ── what "pass" means per outcome ────────────────────────────────────────
def test_expected_status_per_outcome():
    viol = Expect.model_validate({"outcome": "violation", "reason": "clearance"})
    fence = Expect.model_validate({"outcome": "abort", "reason": "fence_breach"})
    done = Expect.model_validate(COMPLETE)
    assert expected_status("clearance", viol) == FAIL and expected_status("fence", viol) == PASS
    assert expected_status("fence", fence) == FAIL and expected_status("clearance", fence) == PASS
    assert {expected_status(i.value, done) for i in done.checked} == {PASS}
