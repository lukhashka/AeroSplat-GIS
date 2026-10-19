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
from aerosplat.mission.models import Code, MissionRefused, PlanningProfile, Severity
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


def test_every_proposed_home_passes_the_planners_checks(tmp_path, street, ground):
    """The point of the search: a proposed home never fails the plan's home or takeoff."""
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap)
    res = search(street, ground, transit_point(target, ap))
    assert res.candidates
    for c in res.candidates:
        plan = plan_mission(target=target, ap=ap, grid=street, occupancy=ms.occupancy_ref(),
                            origin=ms.ORIGIN, by="op:tester", profile=QUIET, home_enu=c.enu,
                            home_source="takeoff search")
        home_or_takeoff = [f for f in plan.findings
                           if f.leg == "takeoff" or f.code.value.startswith("home_")]
        assert home_or_takeoff == [], (c, home_or_takeoff)
        assert plan.home.source == "takeoff search"
        assert plan.transit_z == pytest.approx(c.transit_z, abs=0.51)


def test_transit_point_is_the_entry_point_or_the_pad(tmp_path):
    ap = ms.window_ap()
    target, _, _ = ms.confirmed(tmp_path, ap)
    assert transit_point(target, ap) == pytest.approx(E)   # core: D_entry 12 m, no Δh
    pad = ms.roof_pad_ap()
    pad_target, _, _ = ms.confirmed(tmp_path / "pad", pad)
    assert transit_point(pad_target, pad) == pytest.approx(pad.pad.center)


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
