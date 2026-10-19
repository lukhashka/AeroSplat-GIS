"""Takeoff spots near a target (aerosplat.mission.takeoff) on the synthetic street.

The street of tests/mission_scenes.py: flat ground at z = 0, one box building (x -10..10,
y 15..30, 20 m) with a window on its south wall at (0, 15, 9), the grid x, y in [-40, 40).
A proposed home must be ground, open within the planner's takeoff radius, and pass the
planner's own home and takeoff checks; when none exists, the search must say so and why —
it never relaxes a criterion. The negative cases outnumber the positive ones.
"""

import math

import pytest

import mission_scenes as ms
from aerosplat.mission import planner
from aerosplat.mission.corridor import check_leg
from aerosplat.mission.models import (Code, HandoverOverride, MissionRefused, PlanningProfile,
                                      Severity, Verdict)
from aerosplat.mission.planner import plan_mission
from aerosplat.mission.takeoff import corridor_radius, find_takeoff_spots, transit_point
from aerosplat.surface.models import OCCUPIED, UNKNOWN

QUIET = PlanningProfile(wire_height_m=5.0)                # r = 0.5 + 2 + 2·hypot(0.3, 1.5)
TIGHT = PlanningProfile(wire_height_m=5.0, k_sigma=0.0, r_drone_m=0.3)     # r = 2.3
E = (0.0, 3.0, 9.0)                                       # 12 m in front of the window
GEOREF = 0.3


@pytest.fixture(scope="module")
def street():
    return ms.street_grid()


@pytest.fixture(scope="module")
def ground():
    return ms.street_ground()


def search(grid, ground, target=E, **kw):
    kw.setdefault("profile", QUIET)
    kw.setdefault("sigma_georef_m", GEOREF)
    return find_takeoff_spots(grid, ground, target, **kw)


def with_boxes(*boxes):
    st = ms.base_state()
    for box, value in boxes:
        ms.put(st, box, value)
    return ms.grid_from_state(st)


def clearance_to_box(xy, box) -> float:
    """Horizontal distance from a point to an axis-aligned (x0, x1, y0, y1, ...) box."""
    x0, x1, y0, y1 = box[:4]
    return math.hypot(max(x0 - xy[0], 0.0, xy[0] - x1), max(y0 - xy[1], 0.0, xy[1] - y1))


BUILDING = (-10.0, 10.0, 15.0, 30.0)


# ── positive ─────────────────────────────────────────────────────────────
def test_spots_are_ground_open_ranked_and_apart(street, ground):
    res = search(street, ground)
    r, _ = corridor_radius(QUIET, GEOREF)
    assert res.radius_m == pytest.approx(r) and res.radius_m == pytest.approx(5.559, abs=1e-3)
    c = res.candidates
    assert len(c) == 5 and [k.rank for k in c] == [1, 2, 3, 4, 5]
    assert [k.distance_m for k in c] == sorted(k.distance_m for k in c)
    assert c[0].distance_m < 1.0                           # E itself is open ground
    for a in c:
        assert a.enu[2] == pytest.approx(0.0) and abs(a.above_dtm_m) < 1e-6
        assert clearance_to_box(a.enu, BUILDING) >= r      # no wall within the radius
        assert a.open_radius_m >= r and a.climb_m > 0
        assert min(a.enu[0] + 40, 40 - a.enu[0], a.enu[1] + 40, 40 - a.enu[1]) >= r + 1.0
        for b in c:
            if a is not b:
                assert math.dist(a.enu[:2], b.enu[:2]) >= res.spacing_m
    assert res.best_open is None and "5 takeoff spot(s)" in res.message


def plan_from(target, ap, grid, home, profile=QUIET):
    return plan_mission(target=target, ap=ap, grid=grid, occupancy=ms.occupancy_ref(),
                        origin=ms.ORIGIN, by="op:tester", profile=profile, home_enu=home,
                        home_source="takeoff search")


def search_for(target, ap, grid, ground, profile=QUIET, **kw):
    """The search as the API runs it: towards where the planner's transit ends."""
    end = transit_point(target, ap, grid, profile)
    return search(grid, ground, end.point, profile=profile, approach_note=end.note, **kw)


def test_every_proposed_home_passes_the_planners_checks(tmp_path, street, ground):
    """The point of the search: a proposed home never fails the plan's home or takeoff."""
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap)
    res = search_for(target, ap, street, ground)
    assert res.candidates and res.approach_note == ""
    for c in res.candidates:
        plan = plan_from(target, ap, street, c.enu)
        home_or_takeoff = [f for f in plan.findings
                           if f.leg == "takeoff" or f.code.value.startswith("home_")]
        assert home_or_takeoff == [], (c, home_or_takeoff)
        assert plan.home.source == "takeoff search"
        assert plan.transit_z == pytest.approx(c.transit_z, abs=0.51)


def test_transit_point_is_the_entry_point_or_the_pad(tmp_path, street):
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap)
    end = transit_point(target, ap, street, QUIET)
    assert end.point == pytest.approx(E) and end.note == ""   # core: D_entry 12 m, no Δh
    pad = ms.roof_pad_ap()
    pad_target, _, _ = ms.confirmed(tmp_path / "pad", pad)
    end = transit_point(pad_target, pad, street, QUIET)
    assert end.point == pytest.approx(pad.pad.center) and end.note == ""


# ── the transit ends where the planner's approach puts E ─────────────────
ENTRY50 = {"override": HandoverOverride(entry_distance_m=50.0)}
DIRECT50 = (0.0, -35.0, 9.0)            # 50 m in front of the window: 5 m from the grid's edge


def test_near_the_edge_the_search_aims_at_the_tilted_entry_the_planner_flies(tmp_path, street,
                                                                           ground):
    # E 50 m in front of the window falls 5 m from the grid's edge; it needs r + 2 voxels =
    # 6.56. The planner tilts 15° in the cone (E 6.70 m from the edge). The search aimed at
    # the direct E, where every transit check reaches beyond the grid: "not confirmed, raise
    # max_checks" whatever the budget. Now it aims where the planner's transit ends.
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap, **ENTRY50)
    old = search(street, ground, DIRECT50, max_checks=100)
    assert old.candidates == [] and old.rejected["transit_blocked"] == old.checked == 100
    assert "raise max_checks" in old.message

    end = transit_point(target, ap, street, QUIET)
    assert abs(end.point[0]) == pytest.approx(50 * math.sin(math.radians(15)))
    assert end.point[1] == pytest.approx(15 - 50 * math.cos(math.radians(15)))
    assert ("tilted 15.0° in the 30° cone: the direct approach comes closer to the grid's edge "
            "than r + 2 voxels") in end.note
    res = search_for(target, ap, street, ground, n=3)
    assert len(res.candidates) == 3 and res.target == pytest.approx(end.point)
    assert res.approach_note == end.note and res.message.endswith(
        end.note[0].upper() + end.note[1:])
    for c in res.candidates:
        plan = plan_from(target, ap, street, c.enu)
        assert plan.verdict is Verdict.feasible, (c, plan.findings)
        assert plan.approach.entry == pytest.approx(res.target)
        assert plan.transit_z == pytest.approx(c.transit_z, abs=0.51)


def test_a_blocked_direct_approach_moves_the_transit_end_too(tmp_path, ground):
    # Away from the edge: a lamp post on the direct approach. The planner tilts, so the
    # transit ends at the tilted E, and the search measures and checks towards that one.
    grid = with_boxes(((0.0, 0.5, 4.0, 4.5, 0.0, 20.0), OCCUPIED))
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap)
    end = transit_point(target, ap, grid, TIGHT)
    assert end.point != pytest.approx(E)
    assert "in the 30° cone: the direct approach is blocked" in end.note
    res = search_for(target, ap, grid, ground, profile=TIGHT, n=3)
    assert res.candidates
    for c in res.candidates:
        plan = plan_from(target, ap, grid, c.enu, profile=TIGHT)
        assert plan.approach.entry == pytest.approx(end.point) and plan.approach.tilt_deg > 0
        assert not [f for f in plan.findings if f.leg in ("takeoff", "transit")
                    or f.code.value.startswith("home_")], (c, plan.findings)


def test_with_no_clear_direction_the_search_says_every_plan_is_infeasible(tmp_path, ground):
    # A wall blocks every direction of the cone that fits the grid. The planner shows the
    # least tilted one (15°) and is infeasible from any home; the search checks the transit
    # to that E, and says so rather than proposing homes as if a plan could fly.
    grid = with_boxes(((-20.0, 20.0, 7.0, 7.5, 0.0, 15.0), OCCUPIED))
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap, **ENTRY50)
    end = transit_point(target, ap, grid, QUIET)
    assert ("the approach is blocked in every direction of the 30° cone that fits the grid, so "
            "a plan to this target is infeasible from any home") in end.note
    assert "E of the least tilted (15.0°)" in end.note
    res = search_for(target, ap, grid, ground, n=1)
    assert end.note[1:] in res.message
    for c in res.candidates:
        plan = plan_from(target, ap, grid, c.enu)
        assert plan.verdict is Verdict.infeasible
        assert plan.approach.entry == pytest.approx(end.point)


def test_when_no_direction_fits_the_grid_the_search_refuses_as_the_planner_does(
        tmp_path, street, monkeypatch):
    # E 45 m out with r_app 13.6 m: no direction of the cone keeps E inside the grid, so no
    # home could be planned. Refused at once, with the planner's own reason and hint, before
    # any corridor or home is checked.
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap, override=HandoverOverride(entry_distance_m=45.0,
                                                                         min_clearance_m=10.0))
    calls = []
    monkeypatch.setattr(planner, "check_leg", lambda *a, **k: calls.append(a) or check_leg(*a, **k))
    with pytest.raises(MissionRefused) as e:
        transit_point(target, ap, street, QUIET)
    assert calls == [] and [f.code for f in e.value.findings] == [Code.outside_grid]
    with pytest.raises(MissionRefused) as by_plan:
        plan_from(target, ap, street, (0.0, 0.0, 0.0))
    assert e.value.findings[0].detail == by_plan.value.findings[0].detail
    assert "all 24 other directions" in e.value.findings[0].detail
    assert "occupancy_bbox=" in e.value.findings[0].detail


def test_a_target_off_the_grid_is_refused_before_the_search(tmp_path, street):
    geom = ms.window_ap().geometry.model_copy(update={"center": (0.0, 45.0, 9.0)})
    ap = ms.window_ap(geometry=geom)
    target, _, _ = ms.confirmed(tmp_path, ap)
    with pytest.raises(MissionRefused) as e:
        transit_point(target, ap, street, QUIET)
    assert [f.code for f in e.value.findings] == [Code.outside_grid]
    assert "target (0.0, 45.0)" in e.value.findings[0].detail


def test_a_pad_whose_direct_approach_falls_short_of_the_edge_is_refused_up_front(tmp_path,
                                                                                 street):
    # A transit margin wider than the pad's clearance: r_tr 23.6 m > r_app 5.6 m, so E and H
    # above the pad (17.5 m from the edge) fall short while the pad itself does not. A tilted
    # pad approach depends on each home's transit height, which the search does not model:
    # it says so instead of failing every home's transit.
    wide_transit = PlanningProfile(wire_height_m=5.0, transit_margin_m=20.0)
    pad = ms.roof_pad_ap()
    target, _, _ = ms.confirmed(tmp_path, pad)
    with pytest.raises(MissionRefused) as e:
        transit_point(target, pad, street, wide_transit)
    assert [f.code for f in e.value.findings] == [Code.outside_grid]
    detail = e.value.findings[0].detail
    assert "entry E and hover H above the pad (0.0, 22.5) needs 24.6 m to the edge, has 17.5" in detail
    assert "a tilted one depends on each home's transit height" in detail
    assert "occupancy_bbox=" in detail


# ── negative: what is not a takeoff spot ─────────────────────────────────
def test_a_roof_is_not_a_takeoff_spot(street, ground):
    # The roof is flat, 20 x 15 m: open enough for a tight profile. It is not ground.
    roof = (0.0, 22.0, 20.0)
    res = search(street, ground, roof, profile=TIGHT, max_distance_m=6.0)
    assert res.candidates == [] and res.rejected["building"] > 0
    assert res.message.startswith("no takeoff spot")
    # Without a building map the DTM still tells a roof from the ground.
    bare = search(street, ms.street_ground(labels=False), roof, profile=TIGHT, max_distance_m=6.0)
    assert bare.candidates == [] and bare.rejected["not_ground"] > 0
    # Further out, the street qualifies.
    assert search(street, ground, roof, profile=TIGHT, max_distance_m=20.0).candidates


def test_unknown_space_beside_a_spot_blocks_it(ground):
    blob = (1.0, 2.0, 1.0, 2.0, 0.0, 1.5)                  # nobody saw inside this box
    grid = with_boxes((blob, UNKNOWN))
    res = search(grid, ground)
    assert res.candidates and res.rejected["obstacle_within_radius"] > 0
    for c in res.candidates:
        assert clearance_to_box(c.enu, blob) >= res.radius_m


def test_an_overhang_blocks_the_spot_below_it(ground):
    canopy = (-5.0, 5.0, -5.0, 5.0, 6.0, 6.5)              # open ground under it
    grid = with_boxes((canopy, OCCUPIED))
    res = search(grid, ground)
    for c in res.candidates:
        assert clearance_to_box(c.enu, canopy) >= res.radius_m
    assert all(c.distance_m > 5.0 for c in res.candidates)


def test_a_low_obstacle_within_the_relief_does_not_block(ground):
    # 0.5 m of kerb: lower than the relief the planner ignores at home (0.3 + √3/4 m).
    kerb = (1.0, 2.0, 1.0, 2.0, 0.0, 0.5)
    res = search(with_boxes((kerb, OCCUPIED)), ground)
    assert res.candidates[0].distance_m < 1.0


def test_assumed_columns_are_never_a_home(ground):
    grid = ms.street_grid(assumed=(-40.0, 40.0, -40.0, -20.0))     # free air there is assumed
    res = search(grid, ground, (0.0, -30.0, 0.0), max_distance_m=15.0)
    assert res.rejected["not_measured"] > 0
    assert all(c.enu[1] >= -20.0 for c in res.candidates)


def test_spots_keep_the_planners_distance_from_the_grid_edge(street, ground):
    res = search(street, ground, (30.0, -30.0, 0.0), max_distance_m=50.0)
    assert res.grid_limited and res.rejected["near_grid_edge"] > 0
    edge = res.radius_m + 2 * street.res_m
    for c in res.candidates:
        assert min(40 - c.enu[0], c.enu[1] + 40) >= edge
    assert not search(street, ground, E, max_distance_m=5.0).grid_limited


def test_none_is_reported_honestly_with_what_would_fit(street, ground):
    wide = PlanningProfile(wire_height_m=5.0, sigma_gnss_m=15.0)   # r = 32.5 m on an 80 m grid
    res = search(street, ground, profile=wide)
    assert res.candidates == [] and res.checked == 0
    assert res.message.startswith("no takeoff spot within 150 m of the target")
    assert "r = 32.51 m" in res.message and "σ_GNSS 15" in res.message
    assert res.rejected["near_grid_edge"] + res.rejected["obstacle_within_radius"] > 0
    best = res.best_open
    assert best is not None and best.transit_z is None and best.open_radius_m < res.radius_m
    assert res.sigma_gnss_would_fit_m is not None and res.sigma_gnss_would_fit_m < 15.0
    assert f"σ_GNSS ≤ {res.sigma_gnss_would_fit_m:.2f} m" in res.message
    # ... and the suggestion holds: with that σ the search does find a spot.
    fit = PlanningProfile(wire_height_m=5.0, sigma_gnss_m=0.98 * res.sigma_gnss_would_fit_m)
    assert search(street, ground, profile=fit).candidates


def test_no_sigma_helps_when_the_drone_and_margin_alone_do_not_fit(street, ground):
    huge = PlanningProfile(wire_height_m=5.0, r_drone_m=10.0, transit_margin_m=50.0)
    res = search(street, ground, profile=huge)
    assert res.candidates == [] and res.sigma_gnss_would_fit_m is None
    assert res.best_open is None or "no σ_GNSS would make it qualify" in res.message


def test_the_exact_checks_are_bounded(street, ground):
    res = search(street, ground, n=20, max_checks=2, spacing_m=0.0)
    assert res.checked == 2 and len(res.candidates) == 2
    assert res.rejected["not_checked"] > 0


def test_running_out_of_checks_is_not_reported_as_none(street, ground, monkeypatch):
    from aerosplat.mission import takeoff

    # Every exact check fails (as a home with a blocked probe would): after the budget the
    # search must say it stopped, not that there is no spot.
    monkeypatch.setattr(takeoff, "_verify", lambda *a, **k: ("home_check", math.nan, []))
    res = search(street, ground, max_checks=3)
    assert res.candidates == [] and res.checked == 3 and res.rejected["home_check"] == 3
    assert res.message.startswith("no takeoff spot confirmed")
    assert "raise max_checks" in res.message and res.best_open is None


def test_the_near_miss_is_only_ground_that_lacked_room(street, ground, monkeypatch):
    from aerosplat.mission import takeoff

    # Open ground that fails a later check is no near miss: no σ suggestion may point at it.
    monkeypatch.setattr(takeoff, "_verify", lambda *a, **k: ("transit_leg", 5.0, []))
    res = search(street, ground, max_checks=100_000)
    assert res.candidates == [] and res.rejected["not_checked"] == 0
    need = res.radius_m + 0.25 * street.res_m              # what check_leg asks for
    assert res.best_open is None or res.best_open.open_radius_m < need
    if res.best_open is not None:
        assert res.sigma_gnss_would_fit_m is None or res.sigma_gnss_would_fit_m < 1.5


@pytest.mark.parametrize("kw, match", [
    (dict(target=(float("nan"), 0.0)), "finite"),
    (dict(target=(1.0,)), "finite"),
    (dict(max_distance_m=-1.0), "max_distance_m"),
    (dict(ground_tol_m=float("inf")), "ground_tol_m"),
    (dict(n=0), ">= 1"),
])
def test_bad_input_is_refused(street, ground, kw, match):
    target = kw.pop("target", E)
    with pytest.raises(ValueError, match=match):
        search(street, ground, target, **kw)


def test_a_target_far_from_the_grid_finds_nothing(street, ground):
    res = search(street, ground, (500.0, 500.0, 0.0), max_distance_m=50.0)
    assert res.candidates == [] and res.searched_columns == 0


# ── the planner refuses a home off the grid it would otherwise accept ────
def test_home_search_result_is_inside_the_grid_margin_the_planner_enforces(tmp_path, street,
                                                                           ground):
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap)
    with pytest.raises(MissionRefused) as e:
        plan_mission(target=target, ap=ap, grid=street, occupancy=ms.occupancy_ref(),
                     origin=ms.ORIGIN, by="op:tester", profile=QUIET, home_enu=(38.0, 0.0, 0.0))
    assert [f.code for f in e.value.findings] == [Code.outside_grid]
    assert all(f.severity is Severity.blocking for f in e.value.findings)


# ── command line ─────────────────────────────────────────────────────────
def write_run(root, grid):
    from aerosplat.surface.models import EsdfInfo, OccupancyReport, OccupancyStats
    from aerosplat.surface.occupancy import write_occupancy

    s = grid.state
    stats = OccupancyStats(voxels=int(s.size), free=int((s == 0).sum()),
                           occupied=int((s == 1).sum()), unknown=int((s == 2).sum()),
                           columns={}, observed_fraction=1.0, assumed_fraction=0.0,
                           closed_columns=0, skyline_z=20.0, unknown_ceiling_z=21.0,
                           ceiling_z=grid.spec.z1, n_points=0)
    ms.write_street_semantics(root / "semantics")
    write_occupancy(root / "occupancy", grid, OccupancyReport(
        grid=grid.spec, bounds=grid.spec.bounds(), esdf=EsdfInfo(max_distance_m=20.0),
        stats=stats, inputs={"semantics_dir": str(root / "semantics")}))


def test_cli(tmp_path, street, capsys):
    from aerosplat.mission import takeoff

    write_run(tmp_path, street)
    args = ["--occupancy", str(tmp_path / "occupancy"), "--target", "0,3,9",
            "--sigma-georef", "0.3"]
    assert takeoff.main(args) == 0
    out = capsys.readouterr().out
    assert "takeoff spot(s)" in out and "#1:" in out
    assert takeoff.main(args + ["--sigma-gnss", "15"]) == 1
    assert "no takeoff spot" in capsys.readouterr().out
    assert takeoff.main(args + ["--json", "-n", "2"]) == 0
    assert '"candidates"' in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="semantics"):
        takeoff.main(args + ["--semantics", str(tmp_path / "nowhere")])
