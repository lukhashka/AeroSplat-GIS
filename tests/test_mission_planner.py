"""Approach geometry and corridor feasibility (aerosplat.mission.planner) on synthetic scenes.

Every scene is voxelised by hand (tests/mission_scenes.py), so hover and entry points, radii
and clearances have answers worked out on paper. This module decides whether a drone may fly
somewhere, so the negative cases (DOMAIN §5) outnumber the positive ones: a wall in the
corridor, unknown space on and beside it, something overhead, σ over the gate, a target that
is not confirmed, a home in the wrong place, a transit through unknown columns.
"""

import math

import numpy as np
import pytest

import mission_scenes as ms
from aerosplat.authored.models import Lifecycle
from aerosplat.mission.corridor import Plane, check_leg
from aerosplat.mission.models import (AltitudeFrame, AltitudeFrameError, Code, HandoverOverride,
                                      MissionRefused, PlanningProfile, Severity, Verdict)
from aerosplat.mission import planner
from aerosplat.mission.planner import cone_directions, convex_hull, heading_deg, plan_mission
from aerosplat.surface.models import OCCUPIED, UNKNOWN

QUIET = PlanningProfile(wire_height_m=5.0)               # the window is 9 m up: no wire flag
TIGHT = PlanningProfile(wire_height_m=5.0, k_sigma=0.0, r_drone_m=0.3)


@pytest.fixture(scope="module")
def street():
    return ms.street_grid()


def plan(tmp_path, grid, ap=None, *, profile=QUIET, target_kw=None, **kw):
    ap = ap or ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap, **(target_kw or {}))
    return plan_mission(target=target, ap=ap, grid=grid, occupancy=ms.occupancy_ref(),
                        origin=ms.ORIGIN, by="op:tester", profile=profile, **kw)


def codes(p, severity=None):
    return {f.code for f in p.findings if severity is None or f.severity is severity}


def legs(p):
    return {leg.name: leg for leg in p.legs}


def with_boxes(*boxes):
    st = ms.base_state()
    for box, value in boxes:
        ms.put(st, box, value)
    return ms.grid_from_state(st)


# ── positive: the geometry of §7.2 ───────────────────────────────────────
def test_window_hover_and_entry_points_heading_and_radius(tmp_path, street):
    p = plan(tmp_path, street)
    assert p.verdict is Verdict.feasible, p.findings
    a = p.approach
    # H = c + n·standoff, E = c + n·D_entry (core: 3 m and 12 m), heading −n = north.
    assert a.hover == pytest.approx((0.0, 12.0, 9.0))
    assert a.entry == pytest.approx((0.0, 3.0, 9.0))
    assert a.tilt_deg == 0.0 and a.heading_deg == pytest.approx(0.0)
    # r = r_drone + m + k·σ_total with σ_total² = σ_AP² + σ_georef² + σ_GNSS².
    s_total = math.sqrt(0.1 ** 2 + 0.3 ** 2 + 1.5 ** 2)
    assert p.sigma.approach_m == pytest.approx(s_total)
    assert legs(p)["approach"].radius_m == pytest.approx(0.5 + 1.5 + 2.0 * s_total)
    # Transit has no σ_AP: the vehicle against the map.
    assert legs(p)["transit"].radius_m == pytest.approx(0.5 + 2.0 + 2.0 * math.hypot(0.3, 1.5))
    assert all(leg.min_clearance_m >= leg.radius_m for leg in p.legs)


def test_roof_pad_is_approached_vertically_from_above_the_skyline(tmp_path, street):
    p = plan(tmp_path, street, ms.roof_pad_ap())
    assert p.verdict is Verdict.feasible, p.findings
    a = p.approach
    assert a.kind == "pad" and a.heading_deg is None
    assert a.hover == pytest.approx((0.0, 22.5, 25.0))          # pad centre + up·standoff (5 m)
    assert a.entry[:2] == pytest.approx((0.0, 22.5)) and a.entry[2] == pytest.approx(p.transit_z)
    r_tr = legs(p)["transit"].radius_m
    assert p.transit_z >= 20.0 + r_tr                           # above the roof by the radius
    assert {"takeoff", "transit", "approach", "terminal"} <= set(legs(p))
    assert legs(p)["terminal"].points[-1][2] < 22.0              # down to just above the pad


def test_hover_offset_and_a_more_careful_standoff_move_the_hover_point(tmp_path, street):
    p = plan(tmp_path, street, target_kw={"override": HandoverOverride(standoff_m=4.0)})
    assert p.approach.hover == pytest.approx((0.0, 11.0, 9.0))
    assert p.target.handover.overridden == ["standoff_m"]


def test_cone_search_goes_round_a_pole_in_the_direct_corridor(tmp_path):
    grid = with_boxes(((0.0, 0.5, 4.0, 4.5, 0.0, 20.0), OCCUPIED))     # a lamp post
    p = plan(tmp_path, grid, profile=TIGHT, home_enu=(0.0, -20.0, 0.0))
    assert p.approach.candidate_clearance_m[0][2] == 0.0       # the direct line hits the pole
    assert p.approach.tilt_deg > 0
    assert p.approach.tilt_deg <= 15.0 + 1e-9                  # half of the 30° cone
    assert not [f for f in p.findings if f.leg == "approach"]
    assert legs(p)["approach"].min_clearance_m >= legs(p)["approach"].radius_m
    # The hover point stays `standoff` from the window, on the tilted direction.
    assert np.linalg.norm(np.subtract(p.approach.hover, ms.WINDOW_C)) == pytest.approx(3.0)


def test_transit_climbs_over_a_building_next_to_the_way(tmp_path, street):
    # From the east, the straight line to E passes 2.5 m from the building's corner.
    p = plan(tmp_path, street, home_enu=(22.0, 22.0, 0.0))
    r = legs(p)["transit"].radius_m
    assert p.transit_z >= 20.0 + r
    assert legs(p)["transit"].min_clearance_m >= r
    assert p.verdict is Verdict.feasible, p.findings


def refused_codes(excinfo) -> list[Code]:
    return [f.code for f in excinfo.value.findings]


def test_a_home_near_the_edge_of_the_grid_is_refused_with_a_hint(tmp_path, street):
    # Home 4 m from the grid's edge, takeoff corridor r = 5.56 m: part of the corridor is
    # outside the grid, where nothing is known. That is no obstacle to route around: the
    # mission was never checked, and the operator needs a bigger grid.
    with pytest.raises(MissionRefused) as e:
        plan(tmp_path, street, home_enu=(0.0, 36.0, 0.0))
    assert refused_codes(e) == [Code.outside_grid]
    detail = e.value.findings[0].detail
    assert "home (0.0, 36.0)" in detail and "has 4.0" in detail
    box = [float(v) for v in detail.split("occupancy_bbox=")[1].split(" ")[0].split(",")]
    assert box[0] <= -40 and box[1] <= -40 and box[2] >= 40 and box[3] >= 36 + 6.56


@pytest.mark.parametrize("home", [(100.0, 100.0, 0.0), (0.0, -40.5, 0.0), (39.9, 0.0, 0.0)],
                         ids=["off-map", "just-outside", "on-the-edge"])
def test_a_home_off_the_grid_is_refused(tmp_path, street, home):
    with pytest.raises(MissionRefused) as e:
        plan(tmp_path, street, home_enu=home)
    assert refused_codes(e) == [Code.outside_grid]


def test_a_target_off_the_grid_is_refused(tmp_path, street):
    ap = ms.window_ap(geometry=ms.window_ap().geometry.model_copy(
        update={"center": (0.0, 45.0, 9.0)}))
    with pytest.raises(MissionRefused) as e:
        plan(tmp_path, street, ap)
    assert refused_codes(e) == [Code.outside_grid]
    assert "target (0.0, 45.0)" in e.value.findings[0].detail


def test_an_entry_point_pushed_off_the_grid_is_refused(tmp_path, street):
    # The window is well inside (14 m from the edge), but E is D_entry = 12 m in front of it:
    # 2 m from the edge, inside the corridor radius.
    geom = ms.window_ap().geometry.model_copy(update={"center": (0.0, -26.0, 9.0)})
    with pytest.raises(MissionRefused) as e:
        plan(tmp_path, street, ms.window_ap(geometry=geom))
    assert refused_codes(e) == [Code.outside_grid]
    assert "entry E (0.0, -38.0)" in e.value.findings[0].detail


def test_a_mission_well_inside_the_grid_is_not_refused(tmp_path, street):
    p = plan(tmp_path, street, home_enu=(0.0, -30.0, 0.0))    # 10 m from the edge
    assert Code.outside_grid not in codes(p)


def test_a_cone_that_cannot_stay_inside_the_grid_is_refused_before_any_corridor_check(
        tmp_path, street, monkeypatch):
    # E 45 m in front of the window with an approach corridor of r = 13.6 m: E is 10 m from
    # the edge and needs 14.6, and no direction of the 30° cone brings it closer than 11.5.
    # Checking those wide corridors, which reach beyond the grid, took about two minutes.
    calls = []
    monkeypatch.setattr(planner, "check_leg", lambda *a, **k: calls.append(a) or check_leg(*a, **k))
    with pytest.raises(MissionRefused) as e:
        plan(tmp_path, street, target_kw={
            "override": HandoverOverride(entry_distance_m=45.0, min_clearance_m=10.0)})
    assert refused_codes(e) == [Code.outside_grid] and calls == []
    detail = e.value.findings[0].detail
    assert "entry E (0.0, -30.0) needs 14.6 m to the edge, has 10.0" in detail
    assert "all 24 other directions of the 30° approach cone fall short too" in detail
    assert "occupancy_bbox=" in detail


SQ = math.sqrt(0.5)


def window_at(center, normal=(0.0, -1.0, 0.0)):
    geom = ms.window_ap().geometry.model_copy(update={"center": center, "normal": normal})
    return ms.window_ap(geometry=geom)


def try_all_directions(monkeypatch):
    """The cone search as it was: every direction tried, E and H checked after the choice."""
    monkeypatch.setattr(planner, "directions_inside_grid", lambda grid, directions, *a: directions)


def assert_inside_the_grid(p, grid):
    """E and H of the plan keep the widest corridor radius plus two voxels from the edge, and
    a feasible plan passes every leg check."""
    need = max(legs(p)["transit"].radius_m, legs(p)["approach"].radius_m) + 2 * grid.res_m
    for point in (p.approach.entry, p.approach.hover):
        assert planner.grid_margin_m(grid, point) >= need
    if p.verdict is Verdict.feasible:
        assert all(leg.min_clearance_m >= leg.radius_m for leg in p.legs)


@pytest.mark.parametrize("ap, case", [
    (window_at((0.0, -26.0, 9.0)), "early"),       # E pushed off the grid in every direction
    (window_at((-28.0, 0.0, 9.0), (-SQ, -SQ, 0.0)), "early"),
    (window_at((-26.0, 0.0, 9.0), (-SQ, -SQ, 0.0)), "planned"),  # direct E short, tilted not
    (window_at((-27.0, 0.0, 9.0), (-SQ, -SQ, 0.0)), "planned"),
    (window_at((-27.4, 0.0, 9.0), (-SQ, -SQ, 0.0)), "planned"),
    (ms.window_ap(), "same"),
    (ms.roof_pad_ap(), "same"),
], ids=["entry-pushed-off", "diagonal-28", "diagonal-26", "diagonal-27", "diagonal-27.4",
        "window", "roof-pad"])
def test_only_directions_inside_the_grid_are_tried(tmp_path, street, monkeypatch, ap, case):
    """With no direction of the cone inside the grid the refusal comes before any corridor is
    checked, as the check after the search would give it. Otherwise only directions that fit
    are tried: the old search took the clearest even when its E fell short, and was refused
    outside_grid although a direction that fits had a clear corridor."""
    def outcome(sub):
        try:
            p = plan(tmp_path / sub, street, ap)
        except MissionRefused as e:
            return [f.code for f in e.findings]
        assert_inside_the_grid(p, street)
        return p.verdict, p.approach.direction, p.approach.hover, p.approach.entry, codes(p)

    calls = []
    monkeypatch.setattr(planner, "check_leg", lambda *a, **k: calls.append(a) or check_leg(*a, **k))
    now = outcome("now")
    assert (calls == []) is (case == "early")
    try_all_directions(monkeypatch)
    old = outcome("old")
    if case == "planned":
        assert old == [Code.outside_grid]
        assert now[0] is Verdict.feasible and now[4] == set()
    else:
        assert old == now


def test_a_tilted_direction_that_fits_the_grid_is_planned_near_its_edge(tmp_path, street,
                                                                        monkeypatch):
    # E 50 m in front of the window, at (0, −35): 5 m from the edge, and it needs r + 2 voxels
    # = 6.56. The clearest corridor of the cone puts E at (−8.7, −34.2), 5.8 m from the edge;
    # the 15° tilts towards ±x keep it 6.7 m away.
    entry50 = {"override": HandoverOverride(entry_distance_m=50.0)}
    home = (9.75, 7.75, 0.0)
    p = plan(tmp_path / "now", street, target_kw=entry50, home_enu=home)
    assert p.verdict is Verdict.feasible, p.findings
    a = p.approach
    assert a.tilt_deg == pytest.approx(15.0)
    assert abs(a.entry[0]) == pytest.approx(50 * math.sin(math.radians(15)))
    assert planner.grid_margin_m(street, a.entry) == pytest.approx(6.70, abs=0.01)
    assert a.candidates_outside_grid > 0
    assert a.candidates + a.candidates_outside_grid == 25
    assert_inside_the_grid(p, street)
    # The old search chose the clearest corridor and was refused for its E.
    try_all_directions(monkeypatch)
    with pytest.raises(MissionRefused) as e:
        plan(tmp_path / "old", street, target_kw=entry50, home_enu=home)
    assert refused_codes(e) == [Code.outside_grid]
    assert "entry E (-8.7, -34.2) needs 6.6 m to the edge, has 5.8" in e.value.findings[0].detail


def test_with_no_clear_direction_inside_the_grid_the_plan_is_infeasible_not_refused(
        tmp_path, monkeypatch):
    # The wall blocks every corridor of the cone. The direct E (0, −35) falls short of the
    # grid's edge, so the plan shows the least tilted direction that fits, with its obstacles:
    # a checked plan that is infeasible, not a refusal for a direction never checked.
    grid = with_boxes(((-20.0, 20.0, 7.0, 7.5, 0.0, 15.0), OCCUPIED))
    entry50 = {"override": HandoverOverride(entry_distance_m=50.0)}
    p = plan(tmp_path / "now", grid, target_kw=entry50)
    assert p.verdict is Verdict.infeasible
    a = p.approach
    assert a.tilt_deg == pytest.approx(15.0)                 # the only ring that fits
    assert a.candidates == 8 and a.candidates_outside_grid == 17
    assert_inside_the_grid(p, grid)
    app = [f for f in p.findings if f.leg == "approach" and f.severity is Severity.blocking]
    assert app and app[0].code is Code.obstacle_in_corridor
    assert ("none of the 8 directions in the 30° cone that fit the grid is clear (17 more come "
            "closer to its edge than r + 2 voxels and were not tried), the least tilted (15.0°) "
            "is shown") in app[0].detail
    try_all_directions(monkeypatch)
    with pytest.raises(MissionRefused) as e:
        plan(tmp_path / "old", grid, target_kw=entry50)
    assert refused_codes(e) == [Code.outside_grid]
    assert "entry E (0.0, -35.0)" in e.value.findings[0].detail


def test_wider_radius_is_what_catches_unknown_space_beside_the_corridor(tmp_path):
    # Unknown voxels 3 m beside the axis: outside a 2 m corridor, inside a 5 m one.
    grid = with_boxes(((3.0, 4.0, 7.0, 9.0, 8.0, 10.0), UNKNOWN))
    assert plan(tmp_path, grid, profile=TIGHT).verdict is Verdict.feasible
    wide = plan(tmp_path, grid)
    assert wide.verdict is Verdict.infeasible
    assert Code.unknown_space in codes(wide, Severity.blocking)


# ── negative: corridors ──────────────────────────────────────────────────
def test_a_wall_across_the_corridor_is_infeasible_in_every_direction_of_the_cone(tmp_path):
    grid = with_boxes(((-20.0, 20.0, 7.0, 7.5, 0.0, 15.0), OCCUPIED))
    p = plan(tmp_path, grid)
    assert p.verdict is Verdict.infeasible
    app = [f for f in p.findings if f.leg == "approach"]
    assert app and app[0].code is Code.obstacle_in_corridor and app[0].severity is Severity.blocking
    assert "none of the 25 directions" in app[0].detail
    assert p.approach.candidates == 25                         # 1 + 3 tilts x 8 azimuths


def test_unknown_voxels_on_the_corridor_block_it(tmp_path):
    grid = with_boxes(((-1.0, 1.0, 8.0, 9.0, 8.0, 10.0), UNKNOWN))
    p = plan(tmp_path, grid)
    assert p.verdict is Verdict.infeasible
    assert Code.unknown_space in {f.code for f in p.findings if f.leg == "approach"}


def test_a_balcony_above_the_window_is_an_overhang(tmp_path):
    grid = with_boxes(((-2.0, 2.0, 13.5, 15.0, 11.0, 11.5), OCCUPIED))
    p = plan(tmp_path, grid)
    assert p.verdict is Verdict.infeasible
    assert Code.overhang_above in {f.code for f in p.findings if f.leg == "approach"}


def test_a_cornice_over_the_hover_point_is_an_rtl_risk_not_a_blocker(tmp_path):
    # A slab between home and the target lifts the transit to 18 m; the RTL climb from the
    # hover point then runs into the cornice 5 m above the window.
    grid = with_boxes(((-20.0, 20.0, -10.0, -9.5, 0.0, 15.0), OCCUPIED),
                      ((-5.0, 5.0, 11.0, 15.0, 14.0, 14.5), OCCUPIED))
    p = plan(tmp_path, grid, profile=TIGHT, home_enu=(0.0, -20.0, 0.0))
    assert p.transit_z > 14.5
    risks = [f for f in p.findings if f.leg == "rtl_climb"]
    assert risks and all(f.severity is Severity.risk for f in risks)
    assert Code.overhang_above in {f.code for f in risks}
    assert p.verdict is Verdict.feasible_with_risks, p.findings


def test_transit_through_columns_unknown_to_the_sky_is_infeasible(tmp_path):
    grid = with_boxes(((-3.0, 3.0, -20.0, -18.0, 0.0, 39.0), UNKNOWN))
    p = plan(tmp_path, grid, home_enu=(0.0, -30.0, 0.0))
    assert p.verdict is Verdict.infeasible
    assert Code.unknown_space in {f.code for f in p.findings if f.leg == "transit"}


def test_a_pad_behind_columns_unknown_to_the_sky_is_infeasible_not_a_crash(tmp_path):
    grid = with_boxes(((-3.0, 3.0, 5.0, 7.0, 0.0, 39.0), UNKNOWN))
    p = plan(tmp_path, grid, ms.roof_pad_ap())
    assert p.verdict is Verdict.infeasible
    assert Code.unknown_space in {f.code for f in p.findings if f.leg == "transit"}


def test_a_transit_above_the_ceiling_is_refused(tmp_path, street):
    p = plan(tmp_path, street, profile=PlanningProfile(wire_height_m=5.0, max_alt_rel_m=5.0))
    assert Code.transit_too_high in codes(p, Severity.blocking)
    assert p.verdict is Verdict.infeasible


def test_assumed_free_air_is_a_risk(tmp_path):
    st = ms.base_state()
    grid = ms.grid_from_state(st, assumed=(-8.0, 8.0, 5.0, 13.0))
    p = plan(tmp_path, grid)
    assert Code.assumed_free_air in codes(p, Severity.risk)
    assert p.verdict is Verdict.feasible_with_risks


def test_a_low_window_carries_wire_risk(tmp_path, street):
    p = plan(tmp_path, street, profile=PlanningProfile())     # wires below 15 m: 9 m is low
    assert codes(p) == {Code.wire_risk} and p.verdict is Verdict.feasible_with_risks


def test_a_recorded_power_line_at_a_pad_blocks_it(tmp_path, street):
    p = plan(tmp_path, street, ms.roof_pad_ap(attributes={"hazard": "power_line"}))
    assert Code.wire_risk in codes(p, Severity.blocking)


def test_a_standoff_inside_the_clearance_is_infeasible(tmp_path, street):
    # min_clearance 3.0 (more careful than 1.5) leaves r_drone + m = 3.5 > standoff 3.0.
    p = plan(tmp_path, street, target_kw={"override": HandoverOverride(min_clearance_m=3.0)})
    assert Code.standoff_too_small in codes(p, Severity.blocking)


# ── negative: home ───────────────────────────────────────────────────────
@pytest.mark.parametrize("home, expected", [
    ((0.0, 0.0, 3.0), {Code.home_not_on_surface}),           # floating: altitudes off by 3 m
    ((0.0, 20.0, 10.0), {Code.home_not_on_surface, Code.home_blocked}),   # inside the building
], ids=["floating", "inside"])                               # off the map: refused, see above
def test_home_must_be_on_the_measured_surface(tmp_path, street, home, expected):
    p = plan(tmp_path, street, home_enu=home)
    assert expected <= codes(p, Severity.blocking)
    assert p.verdict is Verdict.infeasible


# ── negative: the target ─────────────────────────────────────────────────
def test_a_draft_target_is_not_planned(tmp_path, street):
    ap = ms.window_ap()
    authored, targets = ms.stores(tmp_path)
    authored.create(ap)
    from aerosplat.taxonomy import load_taxonomy
    draft = targets.create(load_taxonomy(), ap.id, by="tester", sigma_ap_m=0.1,
                           sigma_georef_m=0.3, confirm=False)
    with pytest.raises(MissionRefused) as e:
        plan_mission(target=draft, ap=ap, grid=street, occupancy=ms.occupancy_ref(),
                     origin=ms.ORIGIN, by="op:tester")
    assert [f.code for f in e.value.findings] == [Code.target_not_confirmed]


def test_an_access_point_that_went_stale_after_confirmation_is_not_planned(tmp_path, street):
    ap = ms.window_ap()
    target, authored, _ = ms.confirmed(tmp_path, ap)
    stale = authored.replace(ap.model_copy(update={"lifecycle": Lifecycle.stale}, deep=True))
    with pytest.raises(MissionRefused) as e:
        plan_mission(target=target, ap=stale, grid=street, occupancy=ms.occupancy_ref(),
                     origin=ms.ORIGIN, by="op:tester")
    assert {Code.stale, Code.target_changed} <= {f.code for f in e.value.findings}


def test_sigma_over_the_gate_at_planning_time_is_refused(tmp_path, street):
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap, sigma_georef=1.0)
    # The gate tightened since confirmation (taxonomy edit): planning checks it again.
    tighter = target.model_copy(update={"handover": target.handover.model_copy(
        update={"max_sigma_m": 0.5})}, deep=True)
    with pytest.raises(MissionRefused) as e:
        plan_mission(target=tighter, ap=ap, grid=street, occupancy=ms.occupancy_ref(),
                     origin=ms.ORIGIN, by="op:tester")
    assert [f.code for f in e.value.findings] == [Code.sigma_too_large]


def test_an_amsl_plan_is_refused(tmp_path, street):
    with pytest.raises(AltitudeFrameError, match="unverified"):
        plan(tmp_path, street, altitude_frame=AltitudeFrame.amsl)


# ── the pieces ───────────────────────────────────────────────────────────
def test_masked_check_ignores_only_the_target_surface(street):
    # In front of the south wall at 1 m: the wall counts, unless it is the excluded surface.
    pts = [(0.0, 14.0, 9.0), (0.0, 13.0, 9.0)]
    plane = Plane(point=ms.WINDOW_C, normal=(0.0, -1.0, 0.0), offset_m=0.3 + 0.5 * math.sqrt(3) * 0.5,
                  name="target")
    assert check_leg(street, pts, 2.0).offenders
    assert not check_leg(street, pts, 2.0, plane).offenders
    # ... but the ground is not part of the wall: 1.5 m above it with a 2 m radius is too close.
    low = [(0.0, 10.0, 1.5), (0.0, 12.0, 1.5)]
    assert Code.obstacle_in_corridor in check_leg(street, low, 2.0, plane).offenders


def test_cone_directions_stay_inside_the_cone():
    n = np.array([0.0, -1.0, 0.0])
    dirs = cone_directions(n, 15.0, 3, 8)
    assert len(dirs) == 25 and dirs[0][1] == 0.0
    for a, tilt in dirs:
        assert np.linalg.norm(a) == pytest.approx(1.0)
        assert math.degrees(math.acos(min(1.0, float(a @ n)))) == pytest.approx(tilt, abs=1e-6)
        assert tilt <= 15.0 + 1e-9


def test_heading_and_hull():
    assert heading_deg((0.0, 1.0)) == pytest.approx(0.0)
    assert heading_deg((1.0, 0.0)) == pytest.approx(90.0)
    assert heading_deg((0.0, -1.0)) == pytest.approx(180.0)
    hull = convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float))
    assert len(hull) == 4
